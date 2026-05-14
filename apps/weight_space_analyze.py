"""Post-hoc analysis for the weight-space experiment.

For each (variant, λ):
  - cosine of vec(ΔW_run1) vs vec(ΔW_run2) per module, plus mean,
  - LMC barrier along α·sd1 + (1−α)·sd2 at α ∈ {0, .25, .5, .75, 1},
  - subspace similarity φ(i,j) = ‖U₁[:,:i]ᵀ U₂[:,:j]‖_F² / min(i,j) via SVD of ΔW.

Outputs metrics.json + PNG plots under {input_dir}/{variant}/analysis/
plus cross-variant plots under {input_dir}/plots/.
"""

import argparse
import copy
import json
import os
import queue as _queue
import re
from pathlib import Path

# limit CPU oversubscription — SVD / linear algebra kernels across many
# cores otherwise stall with thread contention on this box.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch
import torch.multiprocessing as mp
torch.set_num_threads(4)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peft import (
    LoraConfig,
    MLoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)

import w2t


LORA_A_RE = re.compile(r"(.+)\.lora_A(?:\.default)?\.weight$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--variants", nargs="+", default=["lora", "mlora", "asym_lora"])
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=(np.linspace(0., 0.5 ** 0.25, 10) ** 2).tolist())
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="LMC interpolation points")
    p.add_argument("--skip_lmc", action="store_true")
    p.add_argument("--with_w2t", action="store_true",
                   help="also compute Weight2Token canonical metrics")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="if set, evaluate LMC grid points in parallel — one "
                        "worker per listed GPU id. CPU-side analysis (cosine, "
                        "φ, W2T) still runs sequentially in the parent. When "
                        "unset, LMC runs sequentially on --device.")
    return p.parse_args()


def build_config(variant, r, lora_alpha):
    if variant == "lora":
        return LoraConfig(task_type="SEQ_CLS", inference_mode=False,
                          r=r, lora_alpha=lora_alpha, lora_dropout=0.1)
    if variant == "asym_lora":
        return LoraConfig(task_type="SEQ_CLS", inference_mode=False,
                          r=r, lora_alpha=lora_alpha, lora_dropout=0.1,
                          use_asym_lora=True)
    return LoraConfig(task_type="SEQ_CLS", inference_mode=False,
                      r=r, lora_alpha=lora_alpha, lora_dropout=0.1,
                      use_mlora=True,
                      mlora_config=MLoraConfig(init_mode="normal",
                                               lr_multiplier=1.0))


def discover_modules(sd):
    mods = []
    for k in sd:
        m = LORA_A_RE.match(k)
        if m:
            mods.append(m.group(1))
    return mods


def b_key(mod, sd):
    for cand in (mod + ".lora_B.weight", mod + ".lora_B.default.weight"):
        if cand in sd:
            return cand
    raise KeyError(f"no lora_B for {mod}")


def a_key(mod, sd):
    for cand in (mod + ".lora_A.weight", mod + ".lora_A.default.weight"):
        if cand in sd:
            return cand
    raise KeyError(f"no lora_A for {mod}")


def delta_w(sd, mod, variant, r, lora_alpha):
    A = sd[a_key(mod, sd)].float()  # (r, in)
    B = sd[b_key(mod, sd)].float()  # (out, r)
    if variant in ("lora", "asym_lora"):
        return (lora_alpha / r) * (B @ A)
    return B @ A  # mLoRA: the multiplicative factor BA


def interp_sd(sd1, sd2, alpha):
    return {k: (1.0 - alpha) * sd1[k] + alpha * sd2[k] for k in sd1}


def tokenize_val_ds(model_name):
    """CPU-only MRPC validation tokenization. Returns (val_ds, tokenizer).

    Doing this in the parent (before spawning workers) avoids races on the
    HuggingFace datasets cache when multiple workers tokenize concurrently.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    datasets = load_dataset("glue", "mrpc")

    def tok(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"],
                         truncation=True, max_length=None)

    tok_ds = datasets["validation"].map(
        tok, batched=True, remove_columns=["idx", "sentence1", "sentence2"])
    tok_ds = tok_ds.rename_column("label", "labels")
    return tok_ds, tokenizer


def make_eval_dl(val_ds, tokenizer, batch_size):
    def collate(ex):
        return tokenizer.pad(ex, padding="longest", return_tensors="pt")
    return DataLoader(val_ds, shuffle=False, collate_fn=collate,
                      batch_size=batch_size)


def tokenize_eval(model_name, batch_size):
    val_ds, tokenizer = tokenize_val_ds(model_name)
    return make_eval_dl(val_ds, tokenizer, batch_size), tokenizer


def eval_model(model, dl, metric, device):
    model.to(device).eval()
    for batch in tqdm(dl, desc="lmc-eval", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        metric.add_batch(predictions=preds, references=batch["labels"])
    return metric.compute()


def cosine_per_module(sd1, sd2, variant, r, lora_alpha):
    out = {}
    for mod in discover_modules(sd1):
        d1 = delta_w(sd1, mod, variant, r, lora_alpha).flatten()
        d2 = delta_w(sd2, mod, variant, r, lora_alpha).flatten()
        if d1.norm() < 1e-12 or d2.norm() < 1e-12:
            out[mod] = float("nan")
        else:
            out[mod] = F.cosine_similarity(d1, d2, dim=0).item()
    return out


def phi_matrix(sd1, sd2, mod, variant, r, lora_alpha):
    """φ(i,j) = ‖U1[:, :i]ᵀ U2[:, :j]‖_F² / min(i,j) for i,j ∈ 1..r.

    ΔW = B·A has rank ≤ r, so only the top-r left singular vectors carry
    signal; we truncate SVD to those.
    """
    d1 = delta_w(sd1, mod, variant, r, lora_alpha)
    d2 = delta_w(sd2, mod, variant, r, lora_alpha)
    U1, _, _ = torch.linalg.svd(d1, full_matrices=False)
    U2, _, _ = torch.linalg.svd(d2, full_matrices=False)
    return w2t.phi_subspace(U1, U2, r)


def _attach_lmc_to_pair_row(pair_row, curve, alphas):
    accs = {a: curve[a]["accuracy"] for a in alphas}
    a0, a1 = alphas[0], alphas[-1]
    mid = alphas[len(alphas) // 2]
    pair_row["lmc_barrier"] = max(accs[a0], accs[a1]) - accs[mid]
    pair_row["lmc_curve"] = {str(a): curve[a] for a in alphas}


def process_pair(variant, lam, pair_idx, pair_dir, args, sample_mod,
                 analysis_dir, eval_dl, metric, model, device, log_prefix=""):
    """End-to-end per-pair analysis: cosine, φ, optional W2T, heatmaps (pair 0
    only), and optional LMC curve. Reuses the caller-provided model for LMC.
    Returns (pair_row, lmc_curve_or_None).
    """
    sd1 = torch.load(pair_dir / "run1" / "adapter.pt", map_location="cpu")
    sd2 = torch.load(pair_dir / "run2" / "adapter.pt", map_location="cpu")

    print(f"\n{log_prefix}[{variant} λ={lam:.3f} pair={pair_idx}] analyzing",
          flush=True)

    pair_row = {"pair": pair_idx}

    # cosine
    cosines = cosine_per_module(sd1, sd2, variant, args.r, args.lora_alpha)
    cos_vals = [v for v in cosines.values() if not np.isnan(v)]
    cos_mean = float(np.mean(cos_vals)) if cos_vals else float("nan")
    pair_row["cosine_mean"] = cos_mean
    pair_row["cosine_per_module"] = cosines

    # subspace similarity φ
    modules = discover_modules(sd1)
    phi_diag_per_mod = []
    for mod in modules:
        phi = phi_matrix(sd1, sd2, mod, variant, args.r, args.lora_alpha)
        phi_diag_per_mod.append(np.diag(phi))
    phi_diag = np.stack(phi_diag_per_mod).mean(axis=0)
    phi_diag_mean = float(phi_diag.mean())
    pair_row["phi_diag_mean"] = phi_diag_mean
    pair_row["phi_diag"] = phi_diag.tolist()

    if pair_idx == 0:
        phi_sample = phi_matrix(sd1, sd2, sample_mod, variant,
                                args.r, args.lora_alpha)
        save_heatmap(
            phi_sample,
            analysis_dir / "plots" / f"phi_heatmap_lambda{lam}.png",
            f"φ(i,j)  {variant} λ={lam:.3f}\n{sample_mod}")

    if args.with_w2t:
        wm = w2t_metrics(sd1, sd2, variant, args.r, args.lora_alpha, sample_mod)
        assert abs(wm["phi_U_diag_mean"] - phi_diag_mean) < 1e-5, (
            f"phi_U_diag_mean {wm['phi_U_diag_mean']} vs legacy "
            f"phi_diag_mean {phi_diag_mean}")
        pair_row["w2t_u_cos_per_rank"] = wm["u_cos_per_rank"].tolist()
        pair_row["w2t_v_cos_per_rank"] = wm["v_cos_per_rank"].tolist()
        pair_row["w2t_sigma_log_ratio_per_rank"] = \
            wm["sigma_log_ratio_per_rank"].tolist()
        pair_row["w2t_phi_U_diag_mean"] = wm["phi_U_diag_mean"]
        pair_row["w2t_phi_V_diag_mean"] = wm["phi_V_diag_mean"]
        pair_row["w2t_sigma_cos"] = wm["sigma_cos"]
        pair_row["w2t_sigma_log_cos"] = wm["sigma_log_cos"]
        pair_row["w2t_sigma_l1"] = wm["sigma_l1"]
        pair_row["w2t_u_cos_mean"] = float(wm["u_cos_per_rank"].mean())
        pair_row["w2t_v_cos_mean"] = float(wm["v_cos_per_rank"].mean())
        if pair_idx == 0:
            save_cos_heatmap(
                wm["u_cos_mat"], wm["modules"],
                analysis_dir / "plots" / f"w2t_u_cos_heatmap_lambda{lam}.png",
                f"u_k cos  {variant} λ={lam:.3f}")
            save_cos_heatmap(
                wm["v_cos_mat"], wm["modules"],
                analysis_dir / "plots" / f"w2t_v_cos_heatmap_lambda{lam}.png",
                f"v_k cos  {variant} λ={lam:.3f}")
            save_heatmap(
                wm["phi_V_sample"],
                analysis_dir / "plots" / f"w2t_phi_V_heatmap_lambda{lam}.png",
                f"φ_V(i,j)  {variant} λ={lam:.3f}\n{sample_mod}")

    curve = None
    if not args.skip_lmc:
        curve = {}
        for a in args.alphas:
            sd = interp_sd(sd1, sd2, a)
            set_peft_model_state_dict(
                model, {k: v.to(device) for k, v in sd.items()})
            em = eval_model(model, eval_dl, metric, device)
            curve[a] = {k: float(v) for k, v in em.items()}
        _attach_lmc_to_pair_row(pair_row, curve, args.alphas)

    print(f"{log_prefix}  pair{pair_idx}: cosine_mean={cos_mean:.4f}  "
          f"phi_diag_mean={phi_diag_mean:.4f}"
          + (f"  barrier={pair_row['lmc_barrier']:.4f}"
             if "lmc_barrier" in pair_row else ""), flush=True)
    return pair_row, curve


def _worker(gpu_id, job_q, result_q, args, variant, val_ds, sample_mod,
            analysis_dir_str, model_name):
    """Per-GPU worker — processes one (lam, pair_idx) per job end-to-end so
    CPU analysis (cosine/φ/W2T) and GPU LMC eval pipeline naturally across
    pairs. The PEFT model is built once and reused across pairs via
    set_peft_model_state_dict."""
    print(f"[gpu{gpu_id}] worker starting (variant={variant})", flush=True)
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    log_prefix = f"[gpu{gpu_id}] "

    if not args.skip_lmc:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        eval_dl = make_eval_dl(val_ds, tokenizer, args.batch_size)
        # experiment_id keeps each worker's evaluate cache from racing
        metric = evaluate.load("glue", "mrpc", experiment_id=f"gpu{gpu_id}")
        cfg = build_config(variant, args.r, args.lora_alpha)
        base = AutoModelForSequenceClassification.from_pretrained(
            model_name, return_dict=True)
        model = get_peft_model(base, copy.deepcopy(cfg))
        model.to(device)
    else:
        eval_dl = metric = model = None

    analysis_dir = Path(analysis_dir_str)
    print(f"{log_prefix}ready, pulling jobs", flush=True)

    while True:
        job = job_q.get()
        if job is None:
            break
        lam, pair_idx, pair_dir_str = job
        pair_row, curve = process_pair(
            variant, lam, pair_idx, Path(pair_dir_str), args, sample_mod,
            analysis_dir, eval_dl, metric, model, device,
            log_prefix=log_prefix)
        result_q.put((lam, pair_idx, pair_row, curve))


def save_heatmap(phi, path, title):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(phi, origin="lower", cmap="viridis", vmin=0, vmax=1,
                   extent=[0.5, phi.shape[1] + 0.5, 0.5, phi.shape[0] + 0.5])
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="φ(i,j)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def plot_metric_vs_lambda(metrics_by_variant, key, ylabel, path, title):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    std_key = key + "_std"
    for variant, rows in metrics_by_variant.items():
        lams = [r["lambda"] for r in rows]
        vals = [r[key] for r in rows]
        if any(std_key in r for r in rows):
            errs = [r.get(std_key, 0.0) for r in rows]
            ax.errorbar(lams, vals, yerr=errs, marker="o", capsize=3,
                        label=variant)
        else:
            ax.plot(lams, vals, marker="o", label=variant)
    ax.set_xlabel("λ")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def save_cos_heatmap(mat, modules, path, title):
    """Heatmap of per-rank cosines — rows are modules, columns are rank k."""
    nm = len(modules)
    fig, ax = plt.subplots(figsize=(max(4.5, mat.shape[1] * 0.4 + 2),
                                    max(3.5, nm * 0.16 + 1.5)))
    im = ax.imshow(mat, origin="lower", cmap="coolwarm",
                   vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("rank index k")
    ax.set_ylabel("module")
    ax.set_title(title)
    ax.set_yticks(range(nm))
    ax.set_yticklabels(modules, fontsize=5)
    plt.colorbar(im, ax=ax, label="cos")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def w2t_metrics(sd1, sd2, variant, r, lora_alpha, sample_mod):
    """Per-(variant, λ) W2T canonical-representation aggregates over modules.

    Canonicalizes each module's ΔW for both runs, sign-aligns pairwise, then
    collects per-rank cosines, φ diagonals on U and V, and spectrum scalars.
    For standard LoRA, ``S`` is scaled by α/r so that singular values refer to
    the same ΔW as ``delta_w``.
    """
    modules = discover_modules(sd1)
    scale = (lora_alpha / r) if variant in ("lora", "asym_lora") else 1.0

    u_cos_rows, v_cos_rows, slr_rows = [], [], []
    phi_U_diag_per_mod, phi_V_diag_per_mod = [], []
    sigma_cos_list, sigma_log_cos_list, sigma_l1_list = [], [], []
    phi_V_sample = None

    for mod in modules:
        A1 = sd1[a_key(mod, sd1)].float()
        B1 = sd1[b_key(mod, sd1)].float()
        A2 = sd2[a_key(mod, sd2)].float()
        B2 = sd2[b_key(mod, sd2)].float()
        U1, S1, V1 = w2t.canonicalize(A1, B1)
        U2, S2, V2 = w2t.canonicalize(A2, B2)
        S1 = S1 * scale
        S2 = S2 * scale

        # Pairwise sign alignment: flip (U2[:,k], V2[:,k]) jointly if sign
        # disagrees with run 1. The joint flip leaves (U2 * S2) @ V2.T unchanged.
        flips = torch.where((U1 * U2).sum(dim=0) < 0,
                            torch.tensor(-1.0), torch.tensor(1.0))
        U2 = U2 * flips
        V2 = V2 * flips

        pr = w2t.per_rank_alignment(U1, S1, V1, U2, S2, V2)
        u_cos_rows.append(pr["u_cos"])
        v_cos_rows.append(pr["v_cos"])
        slr_rows.append(pr["sigma_log_ratio"])

        phi_U = w2t.phi_subspace(U1, U2, r)
        phi_V = w2t.phi_subspace(V1, V2, r)
        phi_U_diag_per_mod.append(np.diag(phi_U))
        phi_V_diag_per_mod.append(np.diag(phi_V))
        if mod == sample_mod:
            phi_V_sample = phi_V

        spec = w2t.sigma_spectrum_metrics(S1, S2)
        sigma_cos_list.append(spec["sigma_cos"])
        sigma_log_cos_list.append(spec["sigma_log_cos"])
        sigma_l1_list.append(spec["sigma_l1"])

    u_cos_mat = np.stack(u_cos_rows)
    v_cos_mat = np.stack(v_cos_rows)
    slr_mat = np.stack(slr_rows)
    phi_U_diag = np.stack(phi_U_diag_per_mod).mean(axis=0)
    phi_V_diag = np.stack(phi_V_diag_per_mod).mean(axis=0)

    return {
        "modules": modules,
        "u_cos_per_rank": u_cos_mat.mean(axis=0),
        "v_cos_per_rank": v_cos_mat.mean(axis=0),
        "sigma_log_ratio_per_rank": slr_mat.mean(axis=0),
        "phi_U_diag_mean": float(phi_U_diag.mean()),
        "phi_V_diag_mean": float(phi_V_diag.mean()),
        "sigma_cos": float(np.mean(sigma_cos_list)),
        "sigma_log_cos": float(np.mean(sigma_log_cos_list)),
        "sigma_l1": float(np.mean(sigma_l1_list)),
        "u_cos_mat": u_cos_mat,
        "v_cos_mat": v_cos_mat,
        "phi_V_sample": phi_V_sample,
    }


def _mean_std(vals):
    """Sample std (ddof=1) for n>=2; std=0 for n=1 (no dispersion to report)."""
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    m = float(arr.mean())
    s = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return m, s


# scalar metric keys that get aggregated into mean/std across pairs.
_SCALAR_KEYS_BASE = ("cosine_mean", "phi_diag_mean")
_SCALAR_KEYS_W2T = (
    "w2t_phi_U_diag_mean", "w2t_phi_V_diag_mean",
    "w2t_sigma_cos", "w2t_sigma_log_cos", "w2t_sigma_l1",
    "w2t_u_cos_mean", "w2t_v_cos_mean",
)
_SCALAR_KEYS_LMC = ("lmc_barrier",)


def aggregate_pair_rows(per_pair, lam, with_w2t, with_lmc):
    """Aggregate per-pair dicts into a single row with mean/std + per-pair lists.

    For each scalar key K in the per-pair dicts we emit K (mean), K_std, and
    K_per_pair. Non-scalar per-pair values (cosine_per_module, phi_diag,
    w2t_*_per_rank, lmc_curve) are kept as lists-of-pairs under K_per_pair.
    """
    row = {"lambda": lam, "n_pairs": len(per_pair)}

    scalar_keys = list(_SCALAR_KEYS_BASE)
    if with_w2t:
        scalar_keys += list(_SCALAR_KEYS_W2T)
    if with_lmc:
        scalar_keys += list(_SCALAR_KEYS_LMC)
    # phi_diag_mean's std key collides grammatically with phi_diag_mean — we
    # namespace it as phi_diag_mean_std (same rule as all other keys: K_std).

    for k in scalar_keys:
        vals = [p[k] for p in per_pair if k in p]
        m, s = _mean_std(vals)
        row[k] = m
        row[k + "_std"] = s
        row[k + "_per_pair"] = vals

    # carry non-scalar per-pair data for downstream consumers
    for k in ("cosine_per_module", "phi_diag",
              "w2t_u_cos_per_rank", "w2t_v_cos_per_rank",
              "w2t_sigma_log_ratio_per_rank", "lmc_curve"):
        vals = [p[k] for p in per_pair if k in p]
        if vals:
            row[k + "_per_pair"] = vals

    return row


def plot_lmc_curves(lmc_per_lambda, variant, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for lam, curve in sorted(lmc_per_lambda.items()):
        alphas = sorted(curve.keys())
        accs = [curve[a]["accuracy"] for a in alphas]
        ax.plot(alphas, accs, marker="o", label=f"λ={lam:.3f}")
    ax.set_xlabel("α (interpolation)")
    ax.set_ylabel("MRPC accuracy")
    ax.set_title(f"LMC curves — {variant}")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    # CPU-only tokenization in the parent — workers (if any) reuse this so
    # they don't race on HuggingFace datasets cache locks.
    val_ds, tokenizer = tokenize_val_ds(args.model_name)

    metrics_by_variant = {}

    for variant in args.variants:
        lmc_per_lambda = {}
        analysis_dir = input_dir / variant / "analysis"
        (analysis_dir / "plots").mkdir(parents=True, exist_ok=True)

        # Build job list (lam, pair_idx, pair_dir) for the variant.
        jobs = []
        pair_count_by_lam = {}
        for lam in args.lambdas:
            lambda_dir = input_dir / variant / f"lambda_{lam}"
            pair_dirs = sorted(
                [d for d in lambda_dir.iterdir()
                 if d.is_dir() and d.name.startswith("pair")],
                key=lambda d: int(d.name[len("pair"):]))
            if not pair_dirs:
                # backward compat: pre-multipair layout with run{1,2}/ directly
                # under lambda_{lam}/
                pair_dirs = [lambda_dir]
            pair_count_by_lam[lam] = len(pair_dirs)
            for pair_idx, pair_dir in enumerate(pair_dirs):
                jobs.append((lam, pair_idx, str(pair_dir)))

        # Pre-pick sample_mod by peeking at any pair's run1 adapter.
        sd_peek = torch.load(Path(jobs[0][2]) / "run1" / "adapter.pt",
                             map_location="cpu")
        sample_mod = discover_modules(sd_peek)[0]
        del sd_peek

        # pair_rows[lam][pair_idx] — fixed-size so out-of-order results land correctly
        pair_rows_by_lam = {lam: [None] * pair_count_by_lam[lam]
                            for lam in args.lambdas}

        if args.gpus:
            ctx = mp.get_context("spawn")
            job_q = ctx.Queue()
            result_q = ctx.Queue()
            for j in jobs:
                job_q.put(j)
            for _ in args.gpus:
                job_q.put(None)

            procs = [ctx.Process(target=_worker,
                                 args=(gpu_id, job_q, result_q, args, variant,
                                       val_ds, sample_mod, str(analysis_dir),
                                       args.model_name))
                     for gpu_id in args.gpus]
            for p in procs:
                p.start()

            received = 0
            while received < len(jobs):
                try:
                    lam, pair_idx, pair_row, curve = result_q.get(timeout=120.0)
                except _queue.Empty:
                    alive = [p for p in procs if p.is_alive()]
                    if not alive:
                        bad = [(p.pid, p.exitcode) for p in procs
                               if p.exitcode not in (0, None)]
                        raise SystemExit(
                            f"all workers exited but only {received}/{len(jobs)} "
                            f"results received; dead workers: {bad}")
                    continue
                pair_rows_by_lam[lam][pair_idx] = pair_row
                if pair_idx == 0 and curve is not None:
                    lmc_per_lambda[lam] = curve
                received += 1

            exit_code = 0
            for p in procs:
                p.join()
                if p.exitcode != 0:
                    exit_code = p.exitcode
            if exit_code != 0:
                raise SystemExit(exit_code)
        else:
            # Sequential path: build model once per variant and reuse it.
            device = args.device if torch.cuda.is_available() else "cpu"
            if not args.skip_lmc:
                eval_dl = make_eval_dl(val_ds, tokenizer, args.batch_size)
                metric = evaluate.load("glue", "mrpc")
                cfg = build_config(variant, args.r, args.lora_alpha)
                base = AutoModelForSequenceClassification.from_pretrained(
                    args.model_name, return_dict=True)
                model = get_peft_model(base, copy.deepcopy(cfg))
                model.to(device)
            else:
                eval_dl = metric = model = None

            for lam, pair_idx, pair_dir_str in jobs:
                pair_row, curve = process_pair(
                    variant, lam, pair_idx, Path(pair_dir_str), args, sample_mod,
                    analysis_dir, eval_dl, metric, model, device)
                pair_rows_by_lam[lam][pair_idx] = pair_row
                if pair_idx == 0 and curve is not None:
                    lmc_per_lambda[lam] = curve

            if model is not None:
                model.to("cpu")
                del model
                torch.cuda.empty_cache()

        # per-lambda aggregation
        rows = []
        for lam in args.lambdas:
            per_pair = pair_rows_by_lam[lam]
            row = aggregate_pair_rows(per_pair, lam, with_w2t=args.with_w2t,
                                      with_lmc=not args.skip_lmc)
            rows.append(row)
            n = row["n_pairs"]
            print(f"[{variant} λ={lam:.3f}] n_pairs={n}  "
                  f"cosine={row['cosine_mean']:.4f}"
                  f"±{row['cosine_mean_std']:.4f}  "
                  f"phi={row['phi_diag_mean']:.4f}"
                  f"±{row['phi_diag_mean_std']:.4f}"
                  + (f"  barrier={row['lmc_barrier']:.4f}"
                     f"±{row['lmc_barrier_std']:.4f}"
                     if "lmc_barrier" in row else ""), flush=True)

        # variant-level outputs
        with (analysis_dir / "metrics.json").open("w") as f:
            json.dump({"variant": variant, "rows": rows}, f, indent=2)

        if not args.skip_lmc and lmc_per_lambda:
            plot_lmc_curves(lmc_per_lambda,
                            variant,
                            analysis_dir / "plots" / f"lmc_curves_{variant}.png")

        metrics_by_variant[variant] = rows

    # cross-variant plots
    plots_dir = input_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_lambda(metrics_by_variant, "cosine_mean",
                          "mean cos(ΔW₁, ΔW₂)",
                          plots_dir / "cosine_vs_lambda.png",
                          "Final-weight cosine vs λ")
    plot_metric_vs_lambda(metrics_by_variant, "phi_diag_mean",
                          "mean diag(φ)",
                          plots_dir / "phi_vs_lambda.png",
                          "Subspace similarity vs λ")
    if not args.skip_lmc:
        plot_metric_vs_lambda(metrics_by_variant, "lmc_barrier",
                              "LMC barrier",
                              plots_dir / "barrier_vs_lambda.png",
                              "LMC barrier vs λ")
    if args.with_w2t:
        plot_metric_vs_lambda(metrics_by_variant, "w2t_phi_V_diag_mean",
                              "mean diag(φ_V)",
                              plots_dir / "w2t_phi_V_vs_lambda.png",
                              "Right-singular-vector subspace similarity vs λ")
        plot_metric_vs_lambda(metrics_by_variant, "w2t_sigma_log_cos",
                              "cos(log(1+σ₁), log(1+σ₂))",
                              plots_dir / "w2t_sigma_log_cos_vs_lambda.png",
                              "Log-spectrum cosine vs λ")
        plot_metric_vs_lambda(metrics_by_variant, "w2t_u_cos_mean",
                              "mean u_k cos",
                              plots_dir / "w2t_u_cos_mean_vs_lambda.png",
                              "Mean per-rank u-cos vs λ")
        plot_metric_vs_lambda(metrics_by_variant, "w2t_v_cos_mean",
                              "mean v_k cos",
                              plots_dir / "w2t_v_cos_mean_vs_lambda.png",
                              "Mean per-rank v-cos vs λ")


if __name__ == "__main__":
    main()
