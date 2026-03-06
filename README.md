# Multiplicative LoRA (mLoRA) — PEFT Extension

This repository extends [HuggingFace PEFT](https://github.com/huggingface/peft) with **Multiplicative LoRA (mLoRA)**, a parameter-efficient fine-tuning method that adapts pretrained weights through element-wise scaling rather than additive residuals.

> **Paper:** [Weight Space Representation Learning via Neural Field Adaptation](https://arxiv.org/abs/2512.01759)

## Standard LoRA vs Multiplicative LoRA

**Standard (additive) LoRA** computes:

```
W' = W + BA
```

where `B ∈ R^{d×r}` and `A ∈ R^{r×k}` are low-rank trainable matrices, and the update `BA` is *added* to the frozen pretrained weight `W`.

**Multiplicative LoRA** computes:

```
W' = W ⊙ BA
```

where `⊙` denotes element-wise (Hadamard) multiplication. Instead of injecting new features additively, mLoRA *scales existing features* of the pretrained weight, preserving channel structure and avoiding feature entanglement.

## Variants

This implementation supports several mLoRA variants via `MLoraConfig`:

| Option | Formula | Description |
|---|---|---|
| Default | `W' = W ⊙ (BA · α/r)` | Direct multiplicative scaling |
| `use_exp=True` | `W' = W ⊙ exp(BA)` | Log-space parameterization; guarantees positive scaling factors |
| `use_weight_norm=True` | `W' = (W ⊙ BA) · ‖W‖_F / ‖W ⊙ BA‖_F` | Frobenius-norm preservation to prevent magnitude drift |
| `fix_a=True` | `W' = W ⊙ (B · 1)` | Freeze A to identity |
| `fix_b=True` | `W' = W ⊙ (1 · A)` | Freeze B to identity |

The `lr_multiplier` parameter allows decoupling the effective learning rate for the adapter weights from the optimizer's global learning rate.

## Installation

Install from this repository:

```bash
pip install -e .
```

## Quickstart

```python
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig
from peft.tuners.lora.config import MLoraConfig

model = AutoModelForSequenceClassification.from_pretrained("roberta-base")

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["query", "value"],
    use_mlora=True,
    mlora_config=MLoraConfig(
        lr_multiplier=10.0,
        use_exp=False,
        use_weight_norm=False,
    ),
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

### With exponential parameterization

```python
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["query", "value"],
    use_mlora=True,
    mlora_config=MLoraConfig(
        lr_multiplier=10.0,
        use_exp=True,        # W' = W * exp(BA)
        use_weight_norm=True, # preserve Frobenius norm
    ),
)
```

## Example applications

Example training scripts are provided in the `apps/` directory:

| Script | Task |
|---|---|
| `apps/image_classification.py` | Image classification |
| `apps/semantic_segmentation.py` | Semantic segmentation |
| `apps/sequence_classification_roberta.py` | Text classification with RoBERTa |
| `apps/dreambooth.py` | DreamBooth fine-tuning |
| `apps/dreambooth_sdxl.py` | DreamBooth with SDXL |
| `apps/dreambooth_wikiart.py` | DreamBooth on WikiArt |

## MLoraConfig reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_exp` | `bool` | `False` | Use exponential parameterization `W' = W * exp(BA)` |
| `use_weight_norm` | `bool` | `False` | Normalize `W'` to match the Frobenius norm of `W` |
| `fix_a` | `bool` | `False` | Fix matrix A to identity (ablation study) |
| `fix_b` | `bool` | `False` | Fix matrix B to identity (ablation study) |
| `lr_multiplier` | `float` | `1.0` | Effective learning rate multiplier for adapter weights |
| `init_mode` | `str` | `"ones"` | Weight initialization mode: `"ones"`, `"normal"`, or `"uniform"` |

## Supported layer types

Multiplicative LoRA is implemented for:
- `nn.Linear` (and HuggingFace `Conv1D`)
- `nn.Embedding`
- `nn.Conv2d` / `nn.Conv3d`

## Code guide

All mLoRA-specific changes live in two files under `src/peft/tuners/lora/`:

### `config.py`

- **`MLoraConfig`** (dataclass) — Holds all mLoRA-specific options (`use_exp`, `use_weight_norm`, `fix_a`, `fix_b`, `lr_multiplier`, `init_mode`).
- **`LoraConfig`** — Extended with two new fields: `use_mlora: bool` and `mlora_config: Optional[MLoraConfig]`.

### `layer.py`

Changes span the three layer classes (`Linear`, `Embedding`, `_ConvNd`) and their shared base `LoraLayer`:

| Location | What changed |
|---|---|
| `LoraLayer.__init__` | Added `use_mlora` / `mlora_config` instance attributes. |
| `LoraLayer.update_layer` | Passes `use_mlora` / `mlora_config` through; calls `mlora_init()` after standard init when mLoRA is enabled. |
| `LoraLayer.mlora_init` | **New method.** Initializes A and B so that the initial scaling factor BA ≈ 1 (identity), meaning the model starts close to the pretrained weights. Supports three modes: `use_exp` (small values so exp ≈ 1), `normal`/`uniform`, and the default `ones`. |
| `Linear.get_delta_weight` | Applies `lr_multiplier`, `fix_a`/`fix_b`, and `use_exp` transforms to A and B before computing the low-rank product BA. In standard LoRA this product is an additive delta; in mLoRA it becomes the element-wise scaling factor. |
| `Linear.forward` | **New branch** (`elif self.use_mlora`): computes `W' = W ⊙ BA` and calls `F.linear(x, W')` directly instead of adding a residual. Optionally preserves the Frobenius norm of W. |
| `Embedding.update_layer` / `Embedding.forward` | Same pattern as Linear — mLoRA config propagation and `W' = W ⊙ BA` applied to the embedding weight table via `F.embedding`. |
| `_ConvNd.update_layer` / `_ConvNd.get_delta_weight` / `_ConvNd.forward` | Same pattern as Linear — mLoRA config propagation, delta weight transforms, and `W' = W ⊙ BA` applied to conv kernels via `F.conv2d`/`F.conv3d`. |

The helper function `dual_normal_init_` at the top of `layer.py` is a utility (not mLoRA-specific) for mixed-distribution weight initialization.

## Citation

```bibtex
@inproceedings{yang2026wsr,
  title     = {Weight Space Representation Learning via Neural Field Adaptation},
  author    = {Yang, Zhuoqian and Salzmann, Mathieu and S{\"u}sstrunk, Sabine},
  booktitle = {Proceedings of the IEEE/CVF Conference on
               Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgments

This project is built on [PEFT](https://github.com/huggingface/peft) by HuggingFace.

```bibtex
@Misc{peft,
  title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
}
```
