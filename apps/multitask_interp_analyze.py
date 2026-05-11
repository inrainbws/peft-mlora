"""Analysis for the multi-task adapter interpolation experiment.

For each variant, load the two task-specific adapters (sd_A, sd_B) saved by
`apps/multitask_interp_experiment.py`, grid-interpolate
   sd_(α,β) = α·sd_A + β·sd_B
   (LoRA A/B matrices use √α/√β instead — variance preserving across the
    bilinear product B·A so that B(α,β)·A(α,β) contributes α·B_1A_1+β·B_2A_2
    on the diagonal)
on a 2-D grid, and evaluate every interpolated adapter on BOTH tasks.
Writes metrics.json + a 3-D accuracy-vs-(α,β) surface plot per (variant,task).
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

import numpy as np
import torch
import torch.multiprocessing as mp

torch.set_num_threads(4)
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

from peft import (
    get_peft_model,
    set_peft_model_state_dict,
)

from multitask_interp_experiment import (
    build_config,
    generate_and_score,
    tokenize_task,
)


def _grid_default():
    # 0.0, 0.1, …, 1.5  (rounded to avoid float-print drift)
    return [round(i * 0.1, 2) for i in range(16)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--variants", nargs="+", default=["lora", "mlora", "asym_lora"])
    p.add_argument("--tasks", nargs=2,
                   default=["financial_phrasebank", "pubmed_qa"])
    p.add_argument("--alphas", type=float, nargs="+", default=_grid_default())
    p.add_argument("--betas", type=float, nargs="+", default=_grid_default())
    p.add_argument("--model_name", type=str, default="google/flan-t5-base")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--max_source_len", type=int, default=384)
    p.add_argument("--max_target_len", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="if set, evaluate grid points in parallel — one worker "
                        "per listed GPU id. When unset, runs sequentially on "
                        "--device.")
    return p.parse_args()


def _is_lora_factor(key):
    return ("lora_A" in key) or ("lora_B" in key)


def interp_sd_grid(sd1, sd2, alpha, beta, variant):
    """Grid interp: α·W_1 + β·W_2 in general; LoRA A/B factors use √α/√β.

    For LoRA layers ΔW = B·A, scaling both factors by √α (resp √β) gives
    α·B_1·A_1 + β·B_2·A_2 + √(αβ)·(B_1·A_2 + B_2·A_1) — the diagonal terms
    match the linear weight-space combination, hence "variance preserving".

    For asym-LoRA, A is frozen and identical across both task runs, so we
    interpolate B linearly (α·B_1 + β·B_2) and leave A unscaled. This gives
    exactly α·B_1·A + β·B_2·A with no cross terms.
    """
    sa = alpha ** 0.5
    sb = beta ** 0.5
    out = {}
    for k in sd1:
        if variant == "asym_lora" and "lora_A" in k:
            # frozen A: identical in sd1 and sd2 — pass through unchanged
            out[k] = sd1[k]
        elif variant == "asym_lora" and "lora_B" in k:
            out[k] = alpha * sd1[k] + beta * sd2[k]
        elif _is_lora_factor(k):
            out[k] = sa * sd1[k] + sb * sd2[k]
        else:
            out[k] = alpha * sd1[k] + beta * sd2[k]
    return out


def make_val_loader(task, tokenizer, max_src, max_tgt, batch_size):
    _, val_ds = tokenize_task(task, tokenizer, max_src, max_tgt)
    return _val_dl_from_ds(val_ds, tokenizer, batch_size)


def _val_dl_from_ds(val_ds, tokenizer, batch_size):
    collator = DataCollatorForSeq2Seq(tokenizer, padding="longest",
                                      label_pad_token_id=-100,
                                      return_tensors="pt")
    return DataLoader(val_ds, shuffle=False, collate_fn=collator,
                      batch_size=batch_size)


def _prepare_val_data(args):
    """Tokenize val splits once in the parent (CPU only) so workers don't
    race on HuggingFace datasets cache locks."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    val_ds_dict = {}
    for t in args.tasks:
        _, val_ds = tokenize_task(t, tokenizer,
                                  args.max_source_len, args.max_target_len)
        val_ds_dict[t] = val_ds
    return val_ds_dict


def eval_base(model_name, val_dls, tokenizer, device, max_new_tokens):
    """Zero-shot reference: evaluate the bare base model (no adapter) on each task."""
    base = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    out = {}
    for task, dl in val_dls.items():
        out[task] = generate_and_score(base, dl, tokenizer, device, max_new_tokens)
        print(f"[base zero-shot task={task}] acc={out[task]:.4f}", flush=True)
    base.to("cpu")
    del base
    torch.cuda.empty_cache()
    return out


def plot_surface(alphas, betas, acc_grid, base_acc, path, title):
    """acc_grid[i, j] = accuracy at (alphas[i], betas[j])."""
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(A, B, acc_grid, cmap="viridis",
                           edgecolor="none", alpha=0.9)
    if base_acc is not None:
        ax.plot_surface(A, B, np.full_like(acc_grid, base_acc),
                        color="gray", alpha=0.15)
    ax.set_xlabel("α  (sd_A weight)")
    ax.set_ylabel("β  (sd_B weight)")
    ax.set_zlabel("accuracy")
    ax.set_zlim(0, 1)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def plot_heatmap(alphas, betas, acc_grid, path, title):
    """2-D heatmap of acc_grid[i, j] over (alphas[i], betas[j])."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    # imshow expects rows=y, cols=x; we want x=α, y=β
    im = ax.imshow(acc_grid.T, origin="lower", cmap="viridis",
                   vmin=0, vmax=1, aspect="auto",
                   extent=[min(alphas), max(alphas),
                           min(betas), max(betas)])
    ax.set_xlabel("α  (sd_A weight)")
    ax.set_ylabel("β  (sd_B weight)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="accuracy")

    # annotate each cell with its accuracy; flip text color for contrast on
    # dark cells (viridis is dark at low values)
    fontsize = max(4, min(8, 60 // max(len(alphas), len(betas))))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            v = acc_grid[i, j]
            if np.isnan(v):
                continue
            ax.text(a, b, f"{v:.2f}",
                    ha="center", va="center", fontsize=fontsize,
                    color="white" if v < 0.5 else "black")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def _eval_base_worker(gpu_id, args, val_ds_dict, result_q):
    """Run eval_base in a child process so the parent stays CUDA-clean.

    Required for the --gpus path: if the parent has initialized CUDA before
    spawning grid-eval workers, the workers can hang.
    """
    print(f"[gpu{gpu_id}] base-eval worker starting", flush=True)
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    val_dls = {t: _val_dl_from_ds(val_ds_dict[t], tokenizer, args.batch_size)
               for t in args.tasks}
    base_acc = eval_base(args.model_name, val_dls, tokenizer, device,
                         args.max_target_len)
    result_q.put(base_acc)


def _worker(gpu_id, job_q, result_q, args, variant, sd_A, sd_B, val_ds_dict):
    print(f"[gpu{gpu_id}] grid worker starting (variant={variant})", flush=True)
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    log_prefix = f"[gpu{gpu_id}] "

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    val_dls = {t: _val_dl_from_ds(val_ds_dict[t], tokenizer, args.batch_size)
               for t in args.tasks}

    cfg = build_config(variant, args.r, args.lora_alpha)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = get_peft_model(base, copy.deepcopy(cfg))
    model.to(device)
    print(f"{log_prefix}ready, pulling jobs", flush=True)

    while True:
        job = job_q.get()
        if job is None:
            break
        i, j, alpha, beta = job
        sd = interp_sd_grid(sd_A, sd_B, alpha, beta, variant)
        set_peft_model_state_dict(
            model, {k: v.to(device) for k, v in sd.items()})
        accs = {}
        for task in args.tasks:
            acc = generate_and_score(model, val_dls[task], tokenizer,
                                     device, args.max_target_len)
            accs[task] = acc
            print(f"{log_prefix}[{variant} α={alpha} β={beta} task={task}] "
                  f"acc={acc:.4f}", flush=True)
        result_q.put((i, j, accs))


def main():
    args = parse_args()
    in_root = Path(args.input_dir)
    in_root.mkdir(parents=True, exist_ok=True)

    if args.gpus:
        # CPU-only pre-tokenize once in the parent so workers don't race on
        # HuggingFace datasets cache locks
        print("[parent] pre-tokenizing val datasets...", flush=True)
        val_ds_dict = _prepare_val_data(args)
        print("[parent] pre-tokenization done", flush=True)

        # keep the parent process CUDA-clean — run eval_base in a child
        ctx = mp.get_context("spawn")
        rq = ctx.Queue()
        p = ctx.Process(target=_eval_base_worker,
                        args=(args.gpus[0], args, val_ds_dict, rq))
        p.start()
        base_acc = rq.get()
        p.join()
        if p.exitcode != 0:
            raise SystemExit(p.exitcode)
        device = None
        tokenizer = None
        val_dls = None
    else:
        val_ds_dict = None
        device = args.device if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        val_dls = {t: make_val_loader(t, tokenizer, args.max_source_len,
                                      args.max_target_len, args.batch_size)
                   for t in args.tasks}
        base_acc = eval_base(args.model_name, val_dls, tokenizer, device,
                             args.max_target_len)

    with (in_root / "base_acc.json").open("w") as f:
        json.dump({"model_name": args.model_name,
                   "tasks": list(args.tasks),
                   "base_acc": base_acc}, f, indent=2)

    alphas = list(args.alphas)
    betas = list(args.betas)

    for variant in args.variants:
        var_dir = in_root / variant
        sd_A = torch.load(var_dir / args.tasks[0] / "adapter.pt",
                          map_location="cpu")
        sd_B = torch.load(var_dir / args.tasks[1] / "adapter.pt",
                          map_location="cpu")

        with (var_dir / args.tasks[0] / "eval.json").open() as f:
            saved_A = json.load(f)["accuracy"]
        with (var_dir / args.tasks[1] / "eval.json").open() as f:
            saved_B = json.load(f)["accuracy"]

        # results[task] is a [len(alphas), len(betas)] array of accuracies
        results = {t: np.full((len(alphas), len(betas)), np.nan)
                   for t in args.tasks}

        if args.gpus:
            ctx = mp.get_context("spawn")
            job_q = ctx.Queue()
            result_q = ctx.Queue()
            n_jobs = 0
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    job_q.put((i, j, alpha, beta))
                    n_jobs += 1
            for _ in args.gpus:
                job_q.put(None)
            procs = [ctx.Process(target=_worker,
                                 args=(gpu_id, job_q, result_q, args,
                                       variant, sd_A, sd_B, val_ds_dict))
                     for gpu_id in args.gpus]
            for p in procs:
                p.start()
            received = 0
            while received < n_jobs:
                try:
                    i, j, accs = result_q.get(timeout=30.0)
                except _queue.Empty:
                    # detect dead workers so we don't hang forever
                    alive = [p for p in procs if p.is_alive()]
                    if not alive:
                        bad = [(p.pid, p.exitcode) for p in procs
                               if p.exitcode not in (0, None)]
                        raise SystemExit(
                            f"all workers exited but only {received}/{n_jobs} "
                            f"results received; dead workers: {bad}")
                    continue
                for task in args.tasks:
                    results[task][i, j] = accs[task]
                received += 1
            exit_code = 0
            for p in procs:
                p.join()
                if p.exitcode != 0:
                    exit_code = p.exitcode
            if exit_code != 0:
                raise SystemExit(exit_code)
        else:
            cfg = build_config(variant, args.r, args.lora_alpha)

            # build the peft-wrapped model once per variant; reload the
            # interpolated state dict in place each grid point.
            base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
            model = get_peft_model(base, copy.deepcopy(cfg))

            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    sd = interp_sd_grid(sd_A, sd_B, alpha, beta, variant)
                    set_peft_model_state_dict(
                        model, {k: v.to(device) for k, v in sd.items()})
                    for task in args.tasks:
                        acc = generate_and_score(model, val_dls[task], tokenizer,
                                                 device, args.max_target_len)
                        results[task][i, j] = acc
                        print(f"[{variant} α={alpha} β={beta} task={task}] "
                              f"acc={acc:.4f}", flush=True)

            model.to("cpu")
            del model, base
            torch.cuda.empty_cache()

        # endpoint sanity: (α=1, β=0) ↔ sd_A, (α=0, β=1) ↔ sd_B
        eps = 1e-3
        if 1.0 in alphas and 0.0 in betas:
            ai = alphas.index(1.0)
            bj = betas.index(0.0)
            ep_A = results[args.tasks[0]][ai, bj]
            if abs(ep_A - saved_A) > eps:
                print(f"WARN [{variant}] (α=1,β=0) acc on {args.tasks[0]} "
                      f"({ep_A:.4f}) != saved ({saved_A:.4f})")
        if 0.0 in alphas and 1.0 in betas:
            ai = alphas.index(0.0)
            bj = betas.index(1.0)
            ep_B = results[args.tasks[1]][ai, bj]
            if abs(ep_B - saved_B) > eps:
                print(f"WARN [{variant}] (α=0,β=1) acc on {args.tasks[1]} "
                      f"({ep_B:.4f}) != saved ({saved_B:.4f})")

        out_dir = var_dir / "analysis"
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)
        with (out_dir / "metrics.json").open("w") as f:
            json.dump({"variant": variant,
                       "alphas": alphas,
                       "betas": betas,
                       "tasks": list(args.tasks),
                       "results": {t: results[t].tolist() for t in args.tasks},
                       "saved_endpoint_acc": {args.tasks[0]: saved_A,
                                              args.tasks[1]: saved_B},
                       "base_acc": base_acc},
                      f, indent=2)
        for task in args.tasks:
            plot_surface(alphas, betas, results[task],
                         base_acc.get(task) if base_acc else None,
                         out_dir / "plots" / f"interp_grid_{task}.png",
                         f"Multi-task grid interp — {variant} / {task}")
            plot_heatmap(alphas, betas, results[task],
                         out_dir / "plots" / f"interp_heatmap_{task}.png",
                         f"Multi-task grid interp — {variant} / {task}")


if __name__ == "__main__":
    main()
