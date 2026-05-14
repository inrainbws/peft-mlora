"""Weight-space structure experiment for LoRA fine-tuning (arXiv 2512.01759 §4.2).

For each (variant, λ), train two adapters starting from variance-preserved-
perturbed inits (seeds seedA, seedB) on GLUE MRPC and save the final state dicts.
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
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

from perturb_init import adapter_keys, perturb_state_dict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["lora", "mlora", "asym_lora"], required=True)
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=(np.linspace(0., 0.5 ** 0.25, 10) ** 2).tolist())
    p.add_argument("--seed_pairs", type=int, nargs="+", default=list(range(20)),
                   help="flat list of seeds, grouped into consecutive pairs "
                        "(length must be even). Each pair is one statistically "
                        "independent replicate; analysis aggregates across them.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="runs/weight_space")
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data_seed", type=int, default=42,
                   help="seed for DataLoader shuffle so both runs share batch order")
    p.add_argument("--no_assert", action="store_true",
                   help="skip init sanity asserts")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="if set, run pair-jobs in parallel — one worker per "
                        "listed GPU id. When unset, runs sequentially on "
                        "--device.")
    return p.parse_args()


def build_config(variant, r, lora_alpha):
    if variant == "lora":
        return LoraConfig(task_type="SEQ_CLS", inference_mode=False,
                          r=r, lora_alpha=lora_alpha, lora_dropout=0.)
    if variant == "mlora":
        return LoraConfig(task_type="SEQ_CLS", inference_mode=False,
                          r=r, lora_alpha=1., lora_dropout=0.,
                          use_mlora=True,
                          mlora_config=MLoraConfig(init_mode="normal"))
    if variant == "asym_lora":
        return LoraConfig(task_type="SEQ_CLS", inference_mode=False,
                          r=r, lora_alpha=lora_alpha, lora_dropout=0.,
                          use_asym_lora=True)
    raise ValueError(variant)


def tokenize_datasets(tokenizer, task="mrpc"):
    datasets = load_dataset("glue", task)

    def tok(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"],
                         truncation=True, max_length=None)

    tok_ds = datasets.map(tok, batched=True,
                          remove_columns=["idx", "sentence1", "sentence2"])
    tok_ds = tok_ds.rename_column("label", "labels")
    return tok_ds


def make_loaders(tok_ds, tokenizer, batch_size, data_seed):
    def collate(ex):
        return tokenizer.pad(ex, padding="longest", return_tensors="pt")

    g = torch.Generator()
    g.manual_seed(data_seed)
    train_dl = DataLoader(tok_ds["train"], shuffle=True, collate_fn=collate,
                          batch_size=batch_size, generator=g)
    eval_dl = DataLoader(tok_ds["validation"], shuffle=False, collate_fn=collate,
                         batch_size=batch_size)
    return train_dl, eval_dl


def init_snapshot(model):
    return {k: v.detach().cpu().clone()
            for k, v in get_peft_model_state_dict(model).items()}


def load_adapter(model, sd, device):
    sd_dev = {k: v.to(device) for k, v in sd.items()}
    set_peft_model_state_dict(model, sd_dev)


def train_and_eval(model, train_dl, eval_dl, epochs, lr, device, metric):
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

    model.eval()
    for batch in tqdm(eval_dl, desc="eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        metric.add_batch(predictions=preds, references=batch["labels"])
    return metric.compute()


def check_init_asserts(init1, init2_raw, init2, lam, keys):
    """Sanity-check perturbation math. Returns diagnostics dict."""
    diag = {"lambda": lam, "per_tensor": {}}
    for k in keys:
        t1 = init1[k].flatten().float()
        t2raw = init2_raw[k].flatten().float()
        tnew = init2[k].flatten().float()
        cos_new_1 = F.cosine_similarity(tnew, t1, dim=0).item()
        cos_raw_1 = F.cosine_similarity(t2raw, t1, dim=0).item()
        diag["per_tensor"][k] = {
            "cos(init2, init1)": cos_new_1,
            "cos(init2_raw, init1)": cos_raw_1,
            "var(init2)": tnew.var().item(),
            "var(init2_raw)": t2raw.var().item(),
        }
    if math.isclose(lam, 0.0):
        for k in keys:
            assert torch.allclose(init2[k], init1[k], atol=1e-6), \
                f"λ=0 should give init2==init1 for {k}"
    if math.isclose(lam, 1.0):
        for k in keys:
            c = diag["per_tensor"][k]["cos(init2, init1)"]
            assert abs(c) < 0.1, f"λ=1 expected |cos|<0.1 for {k}, got {c}"
    if math.isclose(lam, 0.5):
        for k in keys:
            vr = diag["per_tensor"][k]["var(init2_raw)"]
            vn = diag["per_tensor"][k]["var(init2)"]
            rel = abs(vn - vr) / max(vr, 1e-12)
            assert rel < 0.1, f"λ=0.5 variance drift {rel:.3f} for {k}"
    return diag


def process_pair(args, lam, pair_idx, seedA, seedB,
                 device, cfg, tok_ds, tokenizer, metric, log_prefix=""):
    run_dir = (Path(args.output_dir) / args.variant
               / f"lambda_{lam}" / f"pair{pair_idx}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{log_prefix}=== variant={args.variant} λ={lam} "
          f"pair={pair_idx} seeds=({seedA},{seedB}) ===")

    # --- build model1 (seedA) and snapshot init1
    set_seed(seedA)
    base1 = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, return_dict=True)
    model1 = get_peft_model(base1, copy.deepcopy(cfg))
    init1 = init_snapshot(model1)

    # --- build model2 (seedB) and snapshot init2_raw
    set_seed(seedB)
    base2 = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, return_dict=True)
    model2 = get_peft_model(base2, copy.deepcopy(cfg))
    init2_raw = init_snapshot(model2)

    # --- compute perturbed init2
    pkeys = adapter_keys(init1, args.variant)
    init2 = perturb_state_dict(init1, init2_raw, lam, pkeys)

    if not args.no_assert:
        diag = check_init_asserts(init1, init2_raw, init2, lam, pkeys)
        cosines = [d["cos(init2, init1)"]
                   for d in diag["per_tensor"].values()]
        print(f"{log_prefix}  init cos(init2,init1): "
              f"mean={sum(cosines)/len(cosines):.3f} "
              f"min={min(cosines):.3f} max={max(cosines):.3f} "
              f"(over {len(cosines)} tensors)")
        with (run_dir / "init_diagnostics.json").open("w") as f:
            json.dump(diag, f, indent=2)

    # --- load perturbed init into model2
    load_adapter(model2, init2, device)

    # --- set seed for other randomness
    set_seed(0)

    # --- train both runs, shared DataLoader order per (λ, pair)
    for idx, (mdl, seed_used) in enumerate(
            [(model1, seedA), (model2, seedB)], start=1):
        print(f"{log_prefix}  run{idx} (seed={seed_used})")
        train_dl, eval_dl = make_loaders(
            tok_ds, tokenizer, args.batch_size, args.data_seed)
        eval_metric = train_and_eval(
            mdl, train_dl, eval_dl, args.epochs, args.lr, device, metric)
        print(f"{log_prefix}    eval: {eval_metric}")

        out_dir = run_dir / f"run{idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        final_sd = {k: v.detach().cpu()
                    for k, v in get_peft_model_state_dict(mdl).items()}
        torch.save(final_sd, out_dir / "adapter.pt")
        # save init too so analysis can reconstruct ΔW for standard LoRA
        init_snap = init1 if idx == 1 else init2
        torch.save(init_snap, out_dir / "adapter_init.pt")
        with (out_dir / "eval.json").open("w") as f:
            json.dump({"seed": seed_used,
                       "pair": pair_idx,
                       "lambda": lam,
                       "variant": args.variant,
                       **{k: float(v) for k, v in eval_metric.items()}},
                      f, indent=2)

        mdl.to("cpu")
        del mdl
        torch.cuda.empty_cache()

    del base1, base2, model1, model2, init1, init2, init2_raw


def _worker(gpu_id, job_q, args):
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    log_prefix = f"[gpu{gpu_id}] "

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tok_ds = tokenize_datasets(tokenizer)
    metric = evaluate.load("glue", "mrpc", experiment_id=f"gpu{gpu_id}")
    cfg = build_config(args.variant, args.r, args.lora_alpha)

    while True:
        job = job_q.get()
        if job is None:
            break
        lam, pair_idx, seedA, seedB = job
        process_pair(args, lam, pair_idx, seedA, seedB,
                     device, cfg, tok_ds, tokenizer, metric,
                     log_prefix=log_prefix)


def main():
    args = parse_args()

    if len(args.seed_pairs) % 2 != 0 or len(args.seed_pairs) == 0:
        raise ValueError(
            f"--seed_pairs needs an even, nonzero number of ints, "
            f"got {args.seed_pairs}")
    seed_pairs = [(args.seed_pairs[i], args.seed_pairs[i + 1])
                  for i in range(0, len(args.seed_pairs), 2)]

    jobs = [(lam, pair_idx, seedA, seedB)
            for lam in args.lambdas
            for pair_idx, (seedA, seedB) in enumerate(seed_pairs)]

    if args.gpus:
        ctx = mp.get_context("spawn")
        job_q = ctx.Queue()
        for j in jobs:
            job_q.put(j)
        for _ in args.gpus:
            job_q.put(None)

        procs = [ctx.Process(target=_worker, args=(gpu_id, job_q, args))
                 for gpu_id in args.gpus]
        for p in procs:
            p.start()
        exit_code = 0
        for p in procs:
            p.join()
            if p.exitcode != 0:
                exit_code = p.exitcode
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    # --- sequential path
    device = args.device if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tok_ds = tokenize_datasets(tokenizer)
    metric = evaluate.load("glue", "mrpc")
    cfg = build_config(args.variant, args.r, args.lora_alpha)

    for lam, pair_idx, seedA, seedB in jobs:
        process_pair(args, lam, pair_idx, seedA, seedB,
                     device, cfg, tok_ds, tokenizer, metric)


if __name__ == "__main__":
    main()
