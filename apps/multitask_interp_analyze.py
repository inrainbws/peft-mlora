"""Analysis for the multi-task adapter interpolation experiment.

For each variant, load every task-specific adapter saved by
`apps/multitask_interp_experiment.py`. Then for each unordered pair of
tasks (task_A, task_B) from the pool, grid-interpolate in ΔW space:
each LoRA-wrapped base weight is patched to W_orig + scaling·(α·B_1A_1
+ β·B_2A_2) and the LoRA adapter itself is left at default init (B=0)
so it contributes nothing. Evaluate on the pair's two tasks. Writes
one `metrics_<A>__<B>.json` and accuracy-surface + heatmap plots per
(variant, pair, eval-task).
"""

import argparse
import copy
import itertools
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

from peft import get_peft_model

from multitask_interp_experiment import (
    build_config,
    generate_and_score,
    tokenize_task,
    TASKS
)

def _grid_default():
    return [round(-1 + i * 0.25, 2) for i in range(13)]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--variants", nargs="+", default=["lora", "asym_lora"])
    p.add_argument("--tasks", nargs="+",
                   default=list(TASKS.keys()),
                   help="pool of task names; the analyzer iterates over "
                        "all unordered pairs (task_A, task_B)")
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


def _snapshot_base_weights(model):
    """{named_modules_path: W_orig.detach().clone()} for every LoRA layer.

    The path is exactly what `model.named_modules()` returns, which for a
    PEFT-wrapped seq2seq is `base_model.model.<...>.<proj>` (e.g.
    `base_model.model.encoder.block.0.layer.0.SelfAttention.q`).
    """
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "default" in module.lora_A:
            out[name] = module.base_layer.weight.detach().clone()
    return out


def _layer_deltas(sd_A, sd_B, alpha, beta):
    """{named_modules_path: α·B_1A_1 + β·B_2A_2} — caller applies scaling.

    Keys are the LoRA layer's `named_modules` path (matches the snapshot).
    Saved adapter keys look like `base_model.model.<path>.lora_B.default.weight`
    — strip only the trailing `.lora_B.default.weight` to recover <path>.
    """
    deltas = {}
    for k in sd_A:
        if "lora_B" not in k:
            continue
        kA = k.replace("lora_B", "lora_A")
        prefix = k.rsplit(".lora_B", 1)[0]
        B1, A1 = sd_A[k], sd_A[kA]
        B2, A2 = sd_B[k], sd_B[kA]
        deltas[prefix] = alpha * (B1 @ A1) + beta * (B2 @ A2)
    return deltas


def _apply_delta_w(model, sd_A, sd_B, alpha, beta, orig_weights):
    """Patch each LoRA-wrapped base weight to W_orig + scaling·ΔW."""
    deltas = _layer_deltas(sd_A, sd_B, alpha, beta)
    for name, module in model.named_modules():
        if name not in orig_weights:
            continue
        scaling = module.scaling["default"]
        W = module.base_layer.weight
        dw = deltas[name].to(W.device, dtype=W.dtype)
        W.data.copy_(orig_weights[name])
        W.data.add_(scaling * dw)


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


def plot_surface(xs, ys, acc_grid, base_acc, path, title, xlabel, ylabel):
    """acc_grid[i, j] = accuracy at (xs[i], ys[j])."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, acc_grid, cmap="viridis",
                           edgecolor="none", alpha=0.9)
    if base_acc is not None:
        ax.plot_surface(X, Y, np.full_like(acc_grid, base_acc),
                        color="gray", alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("accuracy")
    ax.set_zlim(0, 1)
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)


def plot_heatmap(xs, ys, acc_grid, path, title, xlabel, ylabel):
    """2-D heatmap of acc_grid[i, j] over (xs[i], ys[j])."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    # imshow's extent gives outer cell boundaries — to make cell centers land
    # on the xs/ys values (so text annotations align), pad by half a cell.
    dx = (max(xs) - min(xs)) / (len(xs) - 1) if len(xs) > 1 else 1.0
    dy = (max(ys) - min(ys)) / (len(ys) - 1) if len(ys) > 1 else 1.0
    # imshow expects rows=y, cols=x; acc_grid[i, j] is at (xs[i], ys[j])
    im = ax.imshow(acc_grid.T, origin="lower", cmap="viridis",
                   vmin=0, vmax=1, aspect="auto",
                   extent=[min(xs) - dx / 2, max(xs) + dx / 2,
                           min(ys) - dy / 2, max(ys) + dy / 2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="accuracy")

    # annotate each cell with its accuracy; flip text color for contrast on
    # dark cells (viridis is dark at low values)
    fontsize = max(4, min(8, 60 // max(len(xs), len(ys))))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            v = acc_grid[i, j]
            if np.isnan(v):
                continue
            ax.text(x, y, f"{v:.2f}",
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


def _worker(gpu_id, job_q, result_q, args, variant, sds_by_task, val_ds_dict):
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
    orig_weights = _snapshot_base_weights(model)
    print(f"{log_prefix}ready, pulling jobs", flush=True)

    while True:
        job = job_q.get()
        if job is None:
            break
        task_A, task_B, i, j, alpha, beta = job
        _apply_delta_w(model, sds_by_task[task_A], sds_by_task[task_B],
                       alpha, beta, orig_weights)
        accs = {}
        for task in (task_A, task_B):
            acc = generate_and_score(model, val_dls[task], tokenizer,
                                     device, args.max_target_len)
            accs[task] = acc
            print(f"{log_prefix}[{variant} pair={task_A}__{task_B} "
                  f"α={alpha} β={beta} eval={task}] acc={acc:.4f}",
                  flush=True)
        result_q.put((task_A, task_B, i, j, accs))


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

    pairs = list(itertools.combinations(args.tasks, 2))

    for variant in args.variants:
        var_dir = in_root / variant
        sds_by_task = {t: torch.load(var_dir / t / "adapter.pt",
                                     map_location="cpu")
                       for t in args.tasks}
        saved_endpoint_acc = {}
        for t in args.tasks:
            with (var_dir / t / "eval.json").open() as f:
                saved_endpoint_acc[t] = json.load(f)["accuracy"]

        # pair_results[(A, B)][task] = [len(alphas), len(betas)] grid
        pair_results = {(a, b): {a: np.full((len(alphas), len(betas)), np.nan),
                                 b: np.full((len(alphas), len(betas)), np.nan)}
                        for a, b in pairs}

        if args.gpus:
            ctx = mp.get_context("spawn")
            job_q = ctx.Queue()
            result_q = ctx.Queue()
            n_jobs = 0
            for task_A, task_B in pairs:
                for i, alpha in enumerate(alphas):
                    for j, beta in enumerate(betas):
                        job_q.put((task_A, task_B, i, j, alpha, beta))
                        n_jobs += 1
            for _ in args.gpus:
                job_q.put(None)
            procs = [ctx.Process(target=_worker,
                                 args=(gpu_id, job_q, result_q, args,
                                       variant, sds_by_task, val_ds_dict))
                     for gpu_id in args.gpus]
            for p in procs:
                p.start()
            received = 0
            while received < n_jobs:
                try:
                    tA, tB, i, j, accs = result_q.get(timeout=30.0)
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
                for task, acc in accs.items():
                    pair_results[(tA, tB)][task][i, j] = acc
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

            # build the peft-wrapped model once per variant; patch base
            # weights in place each grid point. Adapter factors stay at
            # default init (lora_B = 0) so the adapter adds nothing.
            base = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
            model = get_peft_model(base, copy.deepcopy(cfg))
            model.to(device)
            orig_weights = _snapshot_base_weights(model)

            for task_A, task_B in pairs:
                sd_A = sds_by_task[task_A]
                sd_B = sds_by_task[task_B]
                results = pair_results[(task_A, task_B)]
                for i, alpha in enumerate(alphas):
                    for j, beta in enumerate(betas):
                        _apply_delta_w(model, sd_A, sd_B, alpha, beta,
                                       orig_weights)
                        for task in (task_A, task_B):
                            acc = generate_and_score(model, val_dls[task],
                                                     tokenizer, device,
                                                     args.max_target_len)
                            results[task][i, j] = acc
                            print(f"[{variant} pair={task_A}__{task_B} "
                                  f"α={alpha} β={beta} eval={task}] "
                                  f"acc={acc:.4f}", flush=True)

            model.to("cpu")
            del model, base
            torch.cuda.empty_cache()

        out_dir = var_dir / "analysis"
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)

        for task_A, task_B in pairs:
            results = pair_results[(task_A, task_B)]
            pair_tag = f"{task_A}__{task_B}"

            # endpoint sanity: (α=1, β=0) ↔ sd_A, (α=0, β=1) ↔ sd_B
            eps = 1e-3
            if 1.0 in alphas and 0.0 in betas:
                ai = alphas.index(1.0)
                bj = betas.index(0.0)
                ep_A = results[task_A][ai, bj]
                if abs(ep_A - saved_endpoint_acc[task_A]) > eps:
                    print(f"WARN [{variant} pair={pair_tag}] (α=1,β=0) acc "
                          f"on {task_A} ({ep_A:.4f}) != saved "
                          f"({saved_endpoint_acc[task_A]:.4f})")
            if 0.0 in alphas and 1.0 in betas:
                ai = alphas.index(0.0)
                bj = betas.index(1.0)
                ep_B = results[task_B][ai, bj]
                if abs(ep_B - saved_endpoint_acc[task_B]) > eps:
                    print(f"WARN [{variant} pair={pair_tag}] (α=0,β=1) acc "
                          f"on {task_B} ({ep_B:.4f}) != saved "
                          f"({saved_endpoint_acc[task_B]:.4f})")

            with (out_dir / f"metrics_{pair_tag}.json").open("w") as f:
                json.dump({"variant": variant,
                           "alphas": alphas,
                           "betas": betas,
                           "task_A": task_A,
                           "task_B": task_B,
                           "results": {t: results[t].tolist()
                                       for t in (task_A, task_B)},
                           "saved_endpoint_acc": {
                               task_A: saved_endpoint_acc[task_A],
                               task_B: saved_endpoint_acc[task_B]},
                           "base_acc": base_acc},
                          f, indent=2)
            for task in (task_A, task_B):
                # put the eval task's adapter weight on X
                if task == task_A:
                    xs, ys, grid = alphas, betas, results[task]
                    xlabel, ylabel = "α  (sd_A weight)", "β  (sd_B weight)"
                else:
                    xs, ys, grid = betas, alphas, results[task].T
                    xlabel, ylabel = "β  (sd_B weight)", "α  (sd_A weight)"
                plot_surface(xs, ys, grid,
                             base_acc.get(task) if base_acc else None,
                             out_dir / "plots" /
                                 f"interp_grid_{pair_tag}__{task}.png",
                             f"Multi-task grid interp — {variant} / "
                             f"{pair_tag} → {task}",
                             xlabel, ylabel)
                plot_heatmap(xs, ys, grid,
                             out_dir / "plots" /
                                 f"interp_heatmap_{pair_tag}__{task}.png",
                             f"Multi-task grid interp — {variant} / "
                             f"{pair_tag} → {task}",
                             xlabel, ylabel)


if __name__ == "__main__":
    main()
