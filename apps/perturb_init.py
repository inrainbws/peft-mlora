"""Helper: variance-preserving perturbation of LoRA init state dicts.

Implements  ι₂(λ) = √(1−λ²)·ι₁ + λ·ε  from arXiv 2512.01759 §4.2, applied
per tensor. Assumes ι₁ and ε are independent iid zero-mean draws from the
same distribution — the caller is responsible for that. For standard LoRA,
only lora_A is zero-mean; lora_B is zeros, so perturbing it is a no-op that
we skip. For mLoRA with init_mode="normal", both A and B are zero-mean.
Asym-LoRA's `lora_A` is also zero-mean (random orthogonal rows from a uniform-random matrix), so perturbing `lora_A` alone (as for standard LoRA) is correct; the perturbed `A` is no longer exactly orthogonal at λ∈(0,1), which is acceptable for this experiment.
"""

import math


def perturb_state_dict(sd1, sd2, lam, keys):
    out = {}
    scale = math.sqrt(1.0 - lam * lam)
    keys = set(keys)
    for k, v in sd1.items():
        if k in keys:
            out[k] = (scale * v + lam * sd2[k]).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def adapter_keys(sd, variant):
    """Return the subset of keys that should be perturbed."""
    a = [k for k in sd if ".lora_A." in k]
    if variant == "lora":
        return a
    if variant == "asym_lora":
        return a
    if variant == "mlora":
        b = [k for k in sd if ".lora_B." in k]
        return a + b
    raise ValueError(f"unknown variant: {variant}")
