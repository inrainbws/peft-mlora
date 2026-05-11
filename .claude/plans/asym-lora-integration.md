# Asymmetric LoRA — implementation plan

Implement "asymmetric LoRA" (arXiv 2402.16842) — LoRA with `A` fixed as a
random orthogonal projection, only `B` trained — and expose it as a third
variant in `apps/weight_space_experiment.py` and
`apps/multitask_interp_experiment.py`.

---

## 1. What the paper does (from the official repo)

`Jiacheng-Zhu-AIML/AsymmetryLoRA → LoRASYM_peft/local_lorasym_all.py`:

```python
# per layer, at construction time
n_w, m_w = self.weight.size()              # base weight shape (out, in)
random_w = torch.rand(n_w, m_w) - 0.5      # uniform in [-0.5, 0.5]
U_rand, S_rand, V_rand = torch.linalg.svd(random_w)   # V_rand is Vh, shape (in, in)
lora_A.weight.copy_(V_rand[:, :r].T)       # (r, in_features), rows orthonormal
nn.init.zeros_(lora_B.weight)              # (out_features, r)
# freeze A
self.lora_A.requires_grad = False
```

Plus a filter in their `_mark_only_adapters_as_trainable` that re-asserts
`requires_grad=False` on any param whose name contains `lora_A`.

Other knobs:
- `A` is **per-layer**, independently sampled — not shared.
- Forward is vanilla additive LoRA: `out += B(A(dropout(x))) * (alpha/r)`.
- No rsLoRA, no PiSSA, no DoRA on top.

---

## 2. Design decisions for this repo

- Mirror the existing `use_mlora` flag pattern. Add a single boolean
  `use_asym_lora` to `LoraConfig`. No `AsymLoraConfig` dataclass — the
  recipe has no knobs. (Easy to promote to a dataclass later if needed.)
- Reuse the existing additive forward path in `Linear.forward` unchanged.
  Asym-LoRA only differs from standard LoRA in how `A` is initialised and
  whether it gets a gradient.
- Support **Linear targets only** (both target backbones —
  `roberta-base` query/value and `flan-t5-base` q/v — are linears). Other
  layer classes (`Embedding`, `Conv1d/2d/3d`, `MultiheadAttention`) absorb
  the new kwarg via `**kwargs` and silently ignore it, matching how mLoRA
  is already partial across layer types.
- The freeze must happen **after** `update_layer`'s trailing
  `self.set_adapter(...)` call, because `set_adapter` does
  `layer.requires_grad_(True)` on every adapter layer.
- `lora_A` for asym-LoRA is *frozen but not zero* (unlike standard
  LoRA where `lora_B` is zero), so the variance-preserving perturbation
  recipe in `apps/perturb_init.py` applies the same way as for the
  standard-LoRA variant (perturb `lora_A` keys only). Caveat: interpolated
  `A` at λ ∈ (0, 1) is no longer exactly orthogonal — only the endpoints
  are. This is acceptable for the weight-space-structure study (the
  experiment is about controlled divergence, not orthogonality of `A`).

---

## 3. Files to change

### 3.1 `src/peft/tuners/lora/config.py`

Add a single new field to `LoraConfig`, next to `use_mlora` / `mlora_config`
(around line 553–560):

```python
use_asym_lora: bool = field(
    default=False,
    metadata={"help": "Whether to use asymmetric LoRA "
                      "(arXiv 2402.16842): A is a frozen random "
                      "orthogonal projection, only B trains."},
)
```

### 3.2 `src/peft/tuners/lora/layer.py`

**(a)** `LoraLayer.__init__` (around line 88):

Add `self.use_asym_lora = False` next to the existing
`self.use_mlora = False` / `self.mlora_config = None`.

**(b)** `LoraLayer.update_layer` signature (line 159–171):

Add a `use_asym_lora=False` keyword arg. Store it on `self`:

```python
self.use_asym_lora = use_asym_lora
```

At the end of `update_layer`, **after** `self.set_adapter(self.active_adapters)`,
add:

```python
if self.use_asym_lora:
    self.asym_lora_init(adapter_name)
```

(Placed after `set_adapter` so the `requires_grad=False` survives.)

**(c)** New method `LoraLayer.asym_lora_init`:

```python
def asym_lora_init(self, adapter_name):
    """Asymmetric LoRA init (arXiv 2402.16842).

    A is set to the first r right-singular vectors of a uniform random
    matrix (rows orthonormal), B is zeroed, and A is frozen.
    """
    A = self.lora_A[adapter_name].weight  # (r, in_features)
    in_features = A.shape[1]
    # base-layer-shape random for parity with the official recipe
    random_w = torch.rand(self.out_features, in_features,
                          device=A.device, dtype=A.dtype) - 0.5
    _, _, Vh = torch.linalg.svd(random_w, full_matrices=True)
    # Vh has shape (in_features, in_features); rows are right singular vectors.
    # Official recipe: V_rand[:, :r].T == first r columns of Vh transposed.
    A.data.copy_(Vh[:, :self.r[adapter_name]].T.contiguous())
    A.requires_grad = False

    nn.init.zeros_(self.lora_B[adapter_name].weight)
    if self.lora_bias[adapter_name]:
        nn.init.zeros_(self.lora_B[adapter_name].bias)
```

Place it next to `mlora_init` (around line 261) for symmetry.

**(d)** `Linear.__init__` (around line 611–644):

Add a `use_asym_lora=False` kwarg and forward it to `update_layer`:

```python
def __init__(
    self,
    ...,
    use_mlora=False,
    mlora_config=None,
    use_asym_lora=False,
    **kwargs,
):
    ...
    self.update_layer(
        adapter_name,
        ...,
        use_mlora=use_mlora,
        mlora_config=mlora_config,
        use_asym_lora=use_asym_lora,
    )
```

**(e)** `dispatch_default` (around line 2075):

```python
kwargs["use_mlora"] = lora_config.use_mlora
kwargs["mlora_config"] = lora_config.mlora_config
kwargs["use_asym_lora"] = lora_config.use_asym_lora
```

Other layer classes (`Embedding`, `Conv1d/2d/3d`, `MultiheadAttention`)
absorb `use_asym_lora` via `**kwargs` and don't propagate it — explicitly
unsupported for now. No change needed to their signatures.

### 3.3 `apps/perturb_init.py`

Extend `adapter_keys` to recognize `"asym_lora"`. For this variant, only
`lora_A` is non-zero at init (B is zero), so perturb A keys only — same as
the standard `"lora"` branch:

```python
def adapter_keys(sd, variant):
    a = [k for k in sd if ".lora_A." in k]
    if variant == "lora":
        return a
    if variant == "asym_lora":
        return a
    if variant == "mlora":
        b = [k for k in sd if ".lora_B." in k]
        return a + b
    raise ValueError(f"unknown variant: {variant}")
```

Update the module docstring to mention asym-LoRA's frozen-A case.

### 3.4 `apps/weight_space_experiment.py`

- `parse_args`: extend `--variant` choices to
  `["lora", "mlora", "asym_lora"]`.
- `build_config`: add an `asym_lora` branch:

  ```python
  if variant == "asym_lora":
      return LoraConfig(task_type="SEQ_CLS", inference_mode=False,
                        r=r, lora_alpha=lora_alpha, lora_dropout=0.,
                        use_asym_lora=True)
  ```

- No change needed to `process_pair`, `train_and_eval`, or the save/load
  logic: asym-LoRA reuses the additive forward path, and
  `get_peft_model_state_dict` / `set_peft_model_state_dict` will round-trip
  both `lora_A` and `lora_B` even though `lora_A` is frozen — this is what
  we want so that re-loading the init for run2 (after perturbation) places
  the perturbed `lora_A` correctly.
- `check_init_asserts`: no change. The `cos`/`var` checks apply since
  asym-LoRA's `lora_A` is zero-mean (each entry of a random orthogonal
  matrix has mean 0).

Output layout becomes
`runs/weight_space/asym_lora/lambda_*/pair*/run{1,2}/...` (mirrors
existing variants).

### 3.5 `apps/multitask_interp_experiment.py`

- `parse_args`: extend `--variant` choices to
  `["lora", "mlora", "asym_lora"]`.
- `build_config`: add the same `asym_lora` branch as above but for
  `task_type="SEQ_2_SEQ_LM"`.
- No other changes. The shared-init flow already saves and reloads the
  full adapter state dict, so the frozen random-orthogonal `A` ends up
  identical in both task runs (good — the analyze script's
  `α·B₁A + β·B₂A` interpolation collapses cleanly to
  `(α·B₁ + β·B₂)·A`).

Output layout becomes
`runs/multitask_interp/asym_lora/{adapter_init.pt, <task>/...}`.

---

## 4. Known follow-ups (out of scope per the task)

These would also need touching if the user later wants the analyze step
to recognise the new variant:

- `apps/weight_space_analyze.py` and `apps/multitask_interp_analyze.py`
  likely hard-code the `{lora, mlora}` variant list. Their
  cross-variant comparison plots and metrics dict would need to be
  extended to glob `asym_lora` too. Not changed in this plan.

- The `apps/README.md` "Quick orientation" table and variant-switch
  paragraph could be updated to mention asym-LoRA. Not changed.

No backbone-side changes (`src/peft/tuners/lora/model.py`,
`peft_model.py`) are needed: `_mark_only_adapters_as_trainable` only
flips non-prefix params off and never re-enables `lora_A`, so the freeze
applied inside `asym_lora_init` survives unless a downstream caller
re-invokes `set_adapter`, which neither experiment script does.
