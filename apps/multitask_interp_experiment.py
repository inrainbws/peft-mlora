"""Multi-task adapter interpolation experiment.

Train N LoRA adapters from the SAME init on a pool of N tasks, save the
final state dicts. `apps/multitask_interp_analyze.py` then linearly
interpolates between every pair of adapters and evaluates each interpolated
point on the pair's two tasks.

Backbone is a seq2seq T5: a single LM head serves all tasks even though
their label vocabularies differ ("positive/negative" vs
"equivalent/not_equivalent" vs ...), so no logit-based classifier head is
needed.
"""

import argparse
import copy
import json
import os
import queue as _queue
from pathlib import Path

# Cap BLAS/OMP threads BEFORE importing torch — with --gpus N, we spawn N
# worker processes; without this each would spawn ncores threads and thrash.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
import torch.multiprocessing as mp

torch.set_num_threads(4)
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
    # Stanford Sentiment Treebank: binary positive/negative sentiment on movie-review sentences (GLUE).
    "sst2": {
        "load": lambda: load_dataset("glue", "sst2"),
        "prepare": lambda ds: (ds["train"], ds["validation"]),
        "format": lambda ex: {
            "src": f"sst2 sentence: {ex['sentence']}",
            "tgt": "positive" if ex["label"] == 1 else "negative",
        },
    },
    # Microsoft Research Paraphrase Corpus: binary paraphrase detection on sentence pairs (GLUE).
    "mrpc": {
        "load": lambda: load_dataset("glue", "mrpc"),
        "prepare": lambda ds: (ds["train"], ds["validation"]),
        "format": lambda ex: {
            "src": (f"mrpc sentence1: {ex['sentence1']} "
                    f"sentence2: {ex['sentence2']}"),
            "tgt": "equivalent" if ex["label"] == 1 else "not_equivalent",
        },
    },
    # Financial Phrasebank-style sentiment on finance-news snippets, 3-way negative/neutral/positive (parquet mirror, pre-split).
    "financial_phrasebank": {
        "load": lambda: load_dataset("nickmuchi/financial-classification"),
        "prepare": lambda ds: (ds["train"], ds["test"]),
        "format": lambda ex: {
            "src": f"financial sentiment: {ex['text']}",
            "tgt": ["negative", "neutral", "positive"][ex["labels"]],
        },
    },
    # PubMedQA (pqa_labeled): 1k expert-annotated biomedical yes/no/maybe QA over PubMed abstracts, single train split.
    "pubmed_qa": {
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
    p.add_argument("--variants", nargs="+",
                   choices=["lora", "asym_lora"],
                   default=["lora", "asym_lora"],
                   help="adapter variants to train; one adapter pool per "
                        "variant, each saved under output_dir/<variant>/")
    p.add_argument("--tasks", nargs="+",
                   default=list(TASKS.keys()),
                   help="pool of task names from TASKS; one adapter is "
                        "trained per task from the shared init")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--model_name", type=str, default="google/flan-t5-base")
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="runs/multitask_interp")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="if set, train each task in parallel — one worker "
                        "per listed GPU id. When unset, runs sequentially on "
                        "--device.")
    p.add_argument("--data_seed", type=int, default=42,
                   help="DataLoader shuffle generator seed")
    p.add_argument("--max_train_examples", type=int, default=5000,
                   help="cap the train split to this many examples; "
                        "0 disables")
    p.add_argument("--per_task_init", action="store_true",
                   help="if set, each task starts from its own random "
                        "init (seed = args.seed + task index) saved at "
                        "<variant>/<task>/adapter_init.pt. Default: all "
                        "tasks share a single init at "
                        "<variant>/adapter_init.pt.")
    return p.parse_args()


def build_config(variant, r, lora_alpha):
    if variant == "lora":
        return LoraConfig(task_type="SEQ_2_SEQ_LM", inference_mode=False,
                          r=r, lora_alpha=lora_alpha, lora_dropout=0.)
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


def _prepare_data(args, tokenizer):
    """Tokenize train+val splits once (CPU only). In --gpus mode this runs in
    the parent so workers don't race on HuggingFace datasets cache locks; in
    sequential mode it hoists tokenization out of the variant loop so the
    same tokenized splits are reused across variants."""
    out = {}
    for t in args.tasks:
        train_ds, val_ds = tokenize_task(t, tokenizer,
                                         args.max_source_len,
                                         args.max_target_len)
        if args.max_train_examples and len(train_ds) > args.max_train_examples:
            train_ds = train_ds.shuffle(seed=0).select(
                range(args.max_train_examples))
        out[t] = (train_ds, val_ds)
    return out


def _train_one_task(task, args, variant, cfg, init, tokenizer, data, device,
                    log_prefix=""):
    """Build a fresh peft model, load the shared init, train+eval on `task`,
    save adapter.pt and eval.json. Shared by sequential and worker paths."""
    out_root = Path(args.output_dir) / variant
    task_dir = out_root / task
    task_dir.mkdir(parents=True, exist_ok=True)

    # rebuild fresh model and explicitly reload the snapshotted init,
    # so all task runs start from identical adapter parameters.
    set_seed(args.seed)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = get_peft_model(base, copy.deepcopy(cfg))
    load_adapter(model, init, device)

    train_ds, val_ds = data
    train_dl, val_dl = make_loaders(train_ds, val_ds, tokenizer,
                                    args.batch_size, args.data_seed)

    train(model, train_dl, args.epochs, args.lr, device)
    acc = generate_and_score(model, val_dl, tokenizer, device,
                             args.max_target_len)
    print(f"{log_prefix}  task={task} accuracy={acc:.4f}", flush=True)

    final_sd = {k: v.detach().cpu()
                for k, v in get_peft_model_state_dict(model).items()}
    torch.save(final_sd, task_dir / "adapter.pt")
    with (task_dir / "eval.json").open("w") as f:
        json.dump({"task": task, "variant": variant,
                   "seed": args.seed, "accuracy": acc}, f, indent=2)

    model.to("cpu")
    del model, base, train_dl, val_dl
    torch.cuda.empty_cache()


def _worker(gpu_id, job_q, result_q, args, variant, init_paths, data_dict):
    print(f"[gpu{gpu_id}] worker starting (variant={variant})", flush=True)
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    log_prefix = f"[gpu{gpu_id}] "

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    cfg = build_config(variant, args.r, args.lora_alpha)
    inits_by_task = {t: torch.load(p, map_location="cpu")
                     for t, p in init_paths.items()}
    print(f"{log_prefix}ready, pulling jobs", flush=True)

    while True:
        job = job_q.get()
        if job is None:
            break
        task = job
        print(f"{log_prefix}=== variant={variant} task={task} "
              f"seed={args.seed} ===", flush=True)
        _train_one_task(task, args, variant, cfg, inits_by_task[task],
                        tokenizer, data_dict[task], device,
                        log_prefix=log_prefix)
        result_q.put(task)


def _snapshot_inits(args, variant, out_root):
    """CPU-only snapshot of the per-variant adapter init(s). Keeps the parent
    CUDA-clean so workers spawned for --gpus don't inherit a broken context.

    With --per_task_init, each task gets its own init from a distinct seed
    (args.seed + task_index), saved under out_root/<task>/adapter_init.pt.
    Otherwise a single init is saved at out_root/adapter_init.pt and shared.

    Returns (cfg, inits, init_paths) where both dicts are keyed by task."""
    cfg = build_config(variant, args.r, args.lora_alpha)
    inits = {}
    init_paths = {}
    if args.per_task_init:
        for i, task in enumerate(args.tasks):
            set_seed(args.seed + i)
            base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
            model = get_peft_model(base, copy.deepcopy(cfg))
            init = init_snapshot(model)
            task_dir = out_root / task
            task_dir.mkdir(parents=True, exist_ok=True)
            path = task_dir / "adapter_init.pt"
            torch.save(init, path)
            inits[task] = init
            init_paths[task] = path
            del model, base
    else:
        set_seed(args.seed)
        base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model = get_peft_model(base, copy.deepcopy(cfg))
        init = init_snapshot(model)
        path = out_root / "adapter_init.pt"
        torch.save(init, path)
        for task in args.tasks:
            inits[task] = init
            init_paths[task] = path
        del model, base
    return cfg, inits, init_paths


def _run_gpus(args, variant, init_paths, data_dict):
    ctx = mp.get_context("spawn")
    job_q = ctx.Queue()
    result_q = ctx.Queue()
    for task in args.tasks:
        job_q.put(task)
    for _ in args.gpus:
        job_q.put(None)
    procs = [ctx.Process(target=_worker,
                         args=(gpu_id, job_q, result_q, args, variant,
                               init_paths, data_dict))
             for gpu_id in args.gpus]
    for p in procs:
        p.start()
    received = 0
    n_jobs = len(args.tasks)
    while received < n_jobs:
        try:
            result_q.get(timeout=1800.0)
        except _queue.Empty:
            alive = [p for p in procs if p.is_alive()]
            if not alive:
                bad = [(p.pid, p.exitcode) for p in procs
                       if p.exitcode not in (0, None)]
                raise SystemExit(
                    f"all workers exited but only {received}/{n_jobs} "
                    f"results received; dead workers: {bad}")
            continue
        received += 1
    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            exit_code = p.exitcode
    if exit_code != 0:
        raise SystemExit(exit_code)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.gpus:
        print("[parent] pre-tokenizing datasets...", flush=True)
    data_dict = _prepare_data(args, tokenizer)
    if args.gpus:
        print("[parent] pre-tokenization done", flush=True)

    device = (args.device if torch.cuda.is_available() else "cpu") \
        if not args.gpus else None

    for variant in args.variants:
        out_root = Path(args.output_dir) / variant
        out_root.mkdir(parents=True, exist_ok=True)
        cfg, inits, init_paths = _snapshot_inits(args, variant, out_root)

        if args.gpus:
            _run_gpus(args, variant, init_paths, data_dict)
        else:
            for task in args.tasks:
                print(f"\n=== variant={variant} task={task} "
                      f"seed={args.seed} ===")
                _train_one_task(task, args, variant, cfg, inits[task],
                                tokenizer, data_dict[task], device)


if __name__ == "__main__":
    main()
