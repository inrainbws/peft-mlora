"""Multi-task adapter interpolation experiment.

Train two LoRA adapters from the SAME init on TWO different tasks (SST-2,
MRPC), save the final state dicts, then `apps/multitask_interp_analyze.py`
linearly interpolates between them and evaluates on both tasks.

Backbone is a seq2seq T5: a single LM head serves both tasks even though
their label vocabularies differ ("positive/negative" vs
"equivalent/not_equivalent"), so no logit-based classifier head is needed.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import (
    LoraConfig,
    MLoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


def _split_train_only(ds, test_size, seed=0):
    s = ds["train"].train_test_split(test_size=test_size, seed=seed)
    return s["train"], s["test"]


TASKS = {
    "sst2": {
        "load": lambda: load_dataset("glue", "sst2"),
        "prepare": lambda ds: (ds["train"], ds["validation"]),
        "format": lambda ex: {
            "src": f"sst2 sentence: {ex['sentence']}",
            "tgt": "positive" if ex["label"] == 1 else "negative",
        },
    },
    "mrpc": {
        "load": lambda: load_dataset("glue", "mrpc"),
        "prepare": lambda ds: (ds["train"], ds["validation"]),
        "format": lambda ex: {
            "src": (f"mrpc sentence1: {ex['sentence1']} "
                    f"sentence2: {ex['sentence2']}"),
            "tgt": "equivalent" if ex["label"] == 1 else "not_equivalent",
        },
    },
    "financial_phrasebank": {
        # Parquet mirror of FPB-style financial sentiment (label 0=neg,
        # 1=neutral, 2=pos). Train/test already split.
        "load": lambda: load_dataset("nickmuchi/financial-classification"),
        "prepare": lambda ds: (ds["train"], ds["test"]),
        "format": lambda ex: {
            "src": f"financial sentiment: {ex['text']}",
            "tgt": ["negative", "neutral", "positive"][ex["labels"]],
        },
    },
    "pubmed_qa": {
        # 1000 expert-annotated biomedical yes/no/maybe QA, single train split.
        "load": lambda: load_dataset("qiaojin/PubMedQA", "pqa_labeled"),
        "prepare": lambda ds: _split_train_only(ds, test_size=0.2),
        "format": lambda ex: {
            "src": (f"pubmedqa question: {ex['question']} "
                    f"context: {' '.join(ex['context']['contexts'])}"),
            "tgt": ex["final_decision"],
        },
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["lora", "mlora", "asym_lora"], required=True)
    p.add_argument("--tasks", nargs=2,
                   default=["financial_phrasebank", "pubmed_qa"],
                   help="exactly two task names from TASKS")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--model_name", type=str, default="google/flan-t5-base")
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="runs/multitask_interp")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data_seed", type=int, default=42,
                   help="DataLoader shuffle generator seed")
    p.add_argument("--max_train_examples", type=int, default=5000,
                   help="cap the train split to this many examples; "
                        "0 disables")
    return p.parse_args()


def build_config(variant, r, lora_alpha):
    if variant == "lora":
        return LoraConfig(task_type="SEQ_2_SEQ_LM", inference_mode=False,
                          r=r, lora_alpha=lora_alpha, lora_dropout=0.)
    if variant == "mlora":
        return LoraConfig(task_type="SEQ_2_SEQ_LM", inference_mode=False,
                          r=r, lora_alpha=1., lora_dropout=0.,
                          use_mlora=True,
                          mlora_config=MLoraConfig(init_mode="normal"))
    if variant == "asym_lora":
        return LoraConfig(task_type="SEQ_2_SEQ_LM", inference_mode=False,
                          r=r, lora_alpha=lora_alpha, lora_dropout=0.,
                          use_asym_lora=True)
    raise ValueError(variant)


def tokenize_task(task_name, tokenizer, max_src, max_tgt):
    """Return (train_ds, val_ds) of {input_ids, attention_mask, labels}."""
    spec = TASKS[task_name]
    ds = spec["load"]()

    def fmt(ex):
        parts = spec["format"](ex)
        src = tokenizer(parts["src"], truncation=True, max_length=max_src)
        tgt_ids = tokenizer(text_target=parts["tgt"], truncation=True,
                            max_length=max_tgt)["input_ids"]
        return {
            "input_ids": src["input_ids"],
            "attention_mask": src["attention_mask"],
            "labels": tgt_ids,
        }

    raw_train, raw_val = spec["prepare"](ds)
    train = raw_train.map(fmt, remove_columns=raw_train.column_names)
    val = raw_val.map(fmt, remove_columns=raw_val.column_names)
    return train, val


def make_loaders(train_ds, val_ds, tokenizer, batch_size, data_seed):
    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest",
                                      label_pad_token_id=-100,
                                      return_tensors="pt")
    g = torch.Generator()
    g.manual_seed(data_seed)
    train_dl = DataLoader(train_ds, shuffle=True, collate_fn=collator,
                          batch_size=batch_size, generator=g)
    val_dl = DataLoader(val_ds, shuffle=False, collate_fn=collator,
                        batch_size=batch_size)
    return train_dl, val_dl


def init_snapshot(model):
    return {k: v.detach().cpu().clone()
            for k, v in get_peft_model_state_dict(model).items()}


def load_adapter(model, sd, device):
    sd_dev = {k: v.to(device) for k, v in sd.items()}
    set_peft_model_state_dict(model, sd_dev)


def train(model, train_dl, epochs, lr, device):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dl) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_dl, desc=f"train e{epoch}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def generate_and_score(model, val_dl, tokenizer, device, max_new_tokens):
    """Greedy decode each batch, exact-match against decoded labels."""
    model.to(device).eval()
    pad_id = tokenizer.pad_token_id
    correct = 0
    total = 0
    for batch in tqdm(val_dl, desc="gen-eval", leave=False):
        gen_in = {"input_ids": batch["input_ids"].to(device),
                  "attention_mask": batch["attention_mask"].to(device)}
        with torch.no_grad():
            out_ids = model.generate(**gen_in, max_new_tokens=max_new_tokens,
                                     num_beams=1, do_sample=False)
        preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        labels = batch["labels"].clone()
        labels[labels == -100] = pad_id
        gold = tokenizer.batch_decode(labels, skip_special_tokens=True)
        for p, t in zip(preds, gold):
            if p.strip() == t.strip():
                correct += 1
            total += 1
    return correct / total if total else float("nan")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    out_root = Path(args.output_dir) / args.variant
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    cfg = build_config(args.variant, args.r, args.lora_alpha)

    # snapshot the shared init once
    set_seed(args.seed)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = get_peft_model(base, copy.deepcopy(cfg))
    init = init_snapshot(model)
    torch.save(init, out_root / "adapter_init.pt")
    del model, base
    torch.cuda.empty_cache()

    # train one adapter per task starting from the snapshotted init
    for task in args.tasks:
        print(f"\n=== variant={args.variant} task={task} seed={args.seed} ===")
        task_dir = out_root / task
        task_dir.mkdir(parents=True, exist_ok=True)

        # rebuild fresh model and explicitly reload the snapshotted init,
        # so both task runs start from identical adapter parameters.
        set_seed(args.seed)
        base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model = get_peft_model(base, copy.deepcopy(cfg))
        load_adapter(model, init, device)

        train_ds, val_ds = tokenize_task(task, tokenizer,
                                         args.max_source_len,
                                         args.max_target_len)
        if args.max_train_examples and len(train_ds) > args.max_train_examples:
            train_ds = train_ds.shuffle(seed=0).select(
                range(args.max_train_examples))
        train_dl, val_dl = make_loaders(train_ds, val_ds, tokenizer,
                                        args.batch_size, args.data_seed)

        train(model, train_dl, args.epochs, args.lr, device)
        acc = generate_and_score(model, val_dl, tokenizer, device,
                                 args.max_target_len)
        print(f"  task={task} accuracy={acc:.4f}")

        final_sd = {k: v.detach().cpu()
                    for k, v in get_peft_model_state_dict(model).items()}
        torch.save(final_sd, task_dir / "adapter.pt")
        with (task_dir / "eval.json").open("w") as f:
            json.dump({"task": task, "variant": args.variant,
                       "seed": args.seed, "accuracy": acc}, f, indent=2)

        model.to("cpu")
        del model, base, train_ds, val_ds, train_dl, val_dl
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
