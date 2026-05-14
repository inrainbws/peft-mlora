# `apps/` — experiment scripts

End-to-end runnable scripts that use this fork's mLoRA against real models / datasets. Two flavors:

- **Training scripts** — fine-tune a backbone with additive LoRA or mLoRA, log to W&B, optionally push to the Hub.
- **Research scripts** — pair an `*_experiment.py` (trains adapters and dumps state dicts to disk) with an `*_analyze.py` (re-loads them and computes metrics / plots). These reproduce paper experiments.

All scripts are invoked from the **repo root** (e.g. `python apps/foo.py …`) and import `peft` from `src/`.

---

## Quick orientation

| Script | Type | Purpose |
|---|---|---|
| `image_classification.py` | training | ViT or ResNet on Food-101 |
| `semantic_segmentation.py` | training | SegFormer or UperNet on `sayeed99/fashion_segmentation` |
| `sequence_classification_roberta.py` | training | RoBERTa-large on GLUE MRPC |
| `dreambooth.py` | training | SD 1.x DreamBooth |
| `dreambooth_sdxl.py` | training | SDXL DreamBooth |
| `dreambooth_wikiart.py` | training | WikiArt-style DreamBooth (W&B baked in) |
| `weight_space_experiment.py` | research | Train two adapters from variance-preserved-perturbed inits, varying λ |
| `weight_space_analyze.py` | research | Cosines / LMC / W2T φ-similarity for the above |
| `multitask_interp_experiment.py` | research | Train two adapters on two tasks from a shared init |
| `multitask_interp_analyze.py` | research | 2-D (α, β) grid interpolation surface + plots |
| `w2t.py` | helper | QR-SVD canonicalization of ΔW = B·A (Weight2Token §3.1) |
| `perturb_init.py` | helper | `ι₂(λ) = √(1−λ²)·ι₁ + λ·ε` per-tensor perturbation |
| `util.py` | helper | wandb segmentation-viz callback, CLIP-I/T and CMMD evaluators |

---

## Pre-reqs

- Install the package editable: `pip install -e .` from repo root.
- A GPU with CUDA (most scripts default `--device cuda`).
- Most scripts call `wandb.login(...)` and `huggingface_hub.login(...)` with **hard-coded tokens** at the top of the file. For your own runs, replace the keys or strip those lines. If you don't, runs will go to the original author's W&B / HF account.
- `report_to='wandb'` is set in `TrainingArguments` for the training scripts. To disable, pass `--report_to none` if exposed, or edit the script.

---

## Training scripts

### `image_classification.py` — ViT / ResNet on Food-101

```bash
python apps/image_classification.py --base_model vit --use_mlora --r 16 --lr_multiplier 10
python apps/image_classification.py --base_model resnet                 # additive LoRA
```

Key flags: `--base_model {vit,resnet}`, `--r`, `--lora_alpha` (auto-defaults to `r` for LoRA, `1` for mLoRA), `--use_mlora`, `--use_exp`, `--use_weight_norm`, `--fix_a`, `--fix_b`, `--mlora_init_mode {ones,normal,uniform}`, `--lr_multiplier`, `--tune_layernorm`.

Target modules: `["query","value"]` (ViT) or `["convolution"]` (ResNet). 50 epochs, 5e-3 LR, fp32. Pushes the trained adapter to `inrainbws/{run_name}` at the end — comment out `lora_model.push_to_hub(...)` if you don't have access.

### `semantic_segmentation.py` — SegFormer / UperNet on fashion segmentation

```bash
python apps/semantic_segmentation.py --base_model segformer --use_mlora --epochs 1
```

Key flags: `--base_model {segformer,upernet}`, plus the standard mLoRA flags. `--num_samples` caps the dataset size.

⚠️ Slight inconsistency: this script passes `normal_init=args.use_normal_init` to `MLoraConfig` (an older field name). The current config uses `init_mode="normal"`. If you hit a `TypeError`, swap to `init_mode="normal" if args.use_normal_init else "ones"`.

Includes a `SegmentationVisualizationCallback` (from `util.py`) that logs prediction overlays to W&B every 100 steps.

### `sequence_classification_roberta.py` — RoBERTa-large on GLUE MRPC

Notebook-style script (jupytext-converted). Constants are hard-coded near the top — edit them in place:

```python
batch_size = 32
model_name_or_path = "roberta-large"
task = "mrpc"
use_mlora = False
num_epochs = 20
```

⚠️ Uses an **older** mLoRA API (`mlora="vanilla", init_lora_weights="ones"` instead of `use_mlora=True, mlora_config=...`). Update before running. Pushes adapter to `inrainbws/roberta-large-peft-{lora,mlora}` at the end.

### `dreambooth.py` / `dreambooth_sdxl.py` / `dreambooth_wikiart.py`

Standard HF DreamBooth scripts patched with mLoRA flags (`--use_mlora`, `--use_exp`, `--use_weight_norm`, `--fix_a`, `--fix_b`, `--lr_multiplier`). They follow upstream `diffusers` conventions, so look at HF's DreamBooth README for `--instance_data_dir`, `--class_prompt`, validation prompts, etc.

`dreambooth_wikiart.py` additionally bakes in W&B integration (`--wandb_key`, `--wandb_project_name`) and uses CLIP-I/T + CMMD evaluators from `util.py`.

---

## Research scripts (paper experiments)

These come in `experiment.py` (run training, dump tensors) + `analyze.py` (compute metrics, render plots) pairs. Outputs go under `runs/<experiment>/<variant>/...`.

### Weight-space structure (arXiv 2512.01759 §4.2)

**What it tests:** if you train two adapters from inits that are continuously interpolated by λ (variance-preserved perturbation `ι₂ = √(1−λ²)·ι₁ + λ·ε`), how does final-weight similarity, subspace overlap, and LMC barrier vary?

```bash
# 1. train (sequential, single GPU)
python apps/weight_space_experiment.py --variant lora  --epochs 3
python apps/weight_space_experiment.py --variant mlora --epochs 3

# 1b. or train in parallel across GPUs
python apps/weight_space_experiment.py --variant lora --gpus 0 1 2 3

# 2. analyze (both variants in one go)
python apps/weight_space_analyze.py --input_dir runs/weight_space --with_w2t
```

Output layout:

```
runs/weight_space/
  {lora,mlora}/
    lambda_{λ}/pair{n}/run{1,2}/{adapter.pt, adapter_init.pt, eval.json}
    analysis/{metrics.json, plots/*.png}
  plots/                       # cross-variant comparison plots
```

Key flags on the experiment side: `--variant {lora,mlora}`, `--lambdas` (default: 10 nonlinearly-spaced points in [0, √0.5]), `--seed_pairs` (flat list, even length, grouped into pairs — each pair is one replicate), `--epochs`, `--r`, `--lora_alpha`, `--gpus`. Default backbone is `roberta-base` on GLUE MRPC.

Key flags on the analysis side: `--with_w2t` enables Weight2Token canonical-form metrics (per-rank u/v cosines, σ-spectrum cosine, φ_U/φ_V), `--skip_lmc` skips the (slow) LMC eval sweep, `--alphas` controls LMC interp points.

### Multi-task adapter interpolation

**What it tests:** train two adapters on two different tasks from the **same** init, then evaluate the 2-D interpolation `α·sd_A + β·sd_B` on a grid against both tasks. Backbone is `flan-t5-base` (single seq2seq head ⇒ no per-task classifier).

```bash
python apps/multitask_interp_experiment.py --variant lora  --tasks financial_phrasebank pubmed_qa
python apps/multitask_interp_experiment.py --variant mlora --tasks financial_phrasebank pubmed_qa

python apps/multitask_interp_analyze.py --input_dir runs/multitask_interp \
    --tasks financial_phrasebank pubmed_qa
```

Pass `--per_task_init` to give each task its own random adapter init (seed = `args.seed + task_index`) instead of the default shared init; the per-task init is then saved at `<variant>/<task>/adapter_init.pt`.

Available tasks (`TASKS` dict in `multitask_interp_experiment.py`): `sst2`, `mrpc`, `financial_phrasebank`, `pubmed_qa`. Add new ones by appending to that dict (`load`, `prepare`, `format` callables).

Output layout:

```
runs/multitask_interp/
  {lora,mlora}/
    adapter_init.pt              # only when sharing inits across tasks (default)
    {task_A,task_B}/{adapter.pt, eval.json}
                                 # + adapter_init.pt here under --per_task_init
    analysis/{metrics.json, plots/interp_{grid,heatmap}_{task}.png}
  base_acc.json                # zero-shot baseline, written by analyze
```

The analysis script uses **variance-preserving** interpolation: regular weights linearly combined as `α·W_1 + β·W_2`, but LoRA A/B factors scaled by `√α` / `√β` so the bilinear product `B·A` contributes diagonally `α·B_1A_1 + β·B_2A_2` (see docstring in `interp_sd_grid`).

---

## Helper modules

- **`w2t.py`** — pure-function library. `canonicalize(A, B) → (U, S, V)` with sign convention; `phi_subspace`, `per_rank_alignment`, `sigma_spectrum_metrics`. Import-only — running it as a script just runs a correctness probe.
- **`perturb_init.py`** — `perturb_state_dict(sd1, sd2, λ, keys)` and `adapter_keys(sd, variant)`. For `variant="lora"` only `lora_A` keys are perturbed (lora_B is zero-init); for `variant="mlora"` (with `init_mode="normal"`) both A and B are perturbed.
- **`util.py`** — `SegmentationVisualizationCallback`, `CLIPEvaluator` (CLIP-I, CLIP-T), `CMMDEvaluator` (polynomial-kernel MMD on CLIP embeddings).

---

## Conventions / patterns

- **Variant switch.** Most scripts build both `additive_lora_config` and `multiplicative_lora_config`, then pick one based on `--use_mlora`. To add a new variant, add a `LoraConfig(..., use_mlora=True, mlora_config=MLoraConfig(...))`.
- **Run naming.** Training scripts compose a `run_name` from rank, alpha, and active mLoRA flags (e.g. `vit_r16_a1_mlora_init_normal_exp_lrm10.0`). This becomes the W&B run name and the HF Hub repo suffix.
- **State-dict round-trips.** Research scripts use `get_peft_model_state_dict` / `set_peft_model_state_dict` to snapshot and reload adapter weights. The init is saved alongside final weights so analysis can reconstruct ΔW for additive LoRA (`B·A` plus init), which matters when `init_mode != "ones"`.
- **Analyze-on-CPU.** Cosines / SVD in `weight_space_analyze.py` run on CPU; only LMC eval moves the interpolated model to GPU. `OMP_NUM_THREADS=4` is set at import time to avoid thread contention on multi-core boxes.

## Adding a new experiment

1. Copy the closest existing pair (`weight_space_*` for replicate-style studies, `multitask_interp_*` for interpolation grids).
2. Reuse `build_config(variant, ...)`, `init_snapshot`, `load_adapter`, `train_and_eval` — they're small and self-contained.
3. Save adapters as `.pt` state dicts under `runs/<your_exp>/<variant>/...`. Mirror the same directory layout so `analyze.py`-style scripts can find them by glob.
4. If you need canonical (gauge-invariant) ΔW representations, import from `w2t.py` instead of re-implementing SVD.
