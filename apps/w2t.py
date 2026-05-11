"""QR-SVD canonicalization of LoRA ΔW = B·A (Weight2Token paper §3.1).

Training-free canonical (U, Σ, V) representation that resolves the GL(r)
gauge symmetry of LoRA factors, plus comparison helpers used by
``weight_space_analyze.py --with_w2t``. No learned parameters.
"""
import numpy as np
import torch


def _apply_sign_convention(U, V):
    """Flip columns of (U, V) jointly so the entry of largest |·| in U[:, k]
    is positive. Tie-break: ``argmax`` returns the first index on ties."""
    r = U.shape[1]
    idx = U.abs().argmax(dim=0)
    pivots = U[idx, torch.arange(r, device=U.device)]
    signs = torch.sign(pivots)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return U * signs, V * signs


def canonicalize(A, B):
    """Canonical SVD of ΔW = B @ A via the QR-SVD trick.

    ``A`` has shape (r, d_in); ``B`` has shape (d_out, r). Returns
    ``(U, S, V)`` with ``U: (d_out, r)``, ``S: (r,)``, ``V: (d_in, r)``
    such that ``(U * S) @ V.T == B @ A`` up to numerical precision.
    """
    A = A.float()
    B = B.float()
    r = B.shape[1]
    assert A.shape[0] == r, f"rank mismatch: A {tuple(A.shape)}, B {tuple(B.shape)}"
    Q_B, R_B = torch.linalg.qr(B, mode="reduced")     # (d_out, r), (r, r)
    Q_A, R_A = torch.linalg.qr(A.T, mode="reduced")   # (d_in, r),  (r, r)
    M = R_B @ R_A.T                                   # (r, r)
    U_M, S, V_Mt = torch.linalg.svd(M, full_matrices=False)
    U = Q_B @ U_M
    V = Q_A @ V_Mt.T
    U, V = _apply_sign_convention(U, V)
    return U, S, V


def per_rank_alignment(U1, S1, V1, U2, S2, V2):
    """Per-rank cosines and log-spectrum ratio. Assumes (U2, V2) have already
    been pairwise sign-aligned against (U1, V1)."""
    u_cos = (U1 * U2).sum(dim=0)
    v_cos = (V1 * V2).sum(dim=0)
    sigma_log_ratio = torch.log1p(S1) - torch.log1p(S2)
    return {
        "u_cos": u_cos.detach().cpu().numpy(),
        "v_cos": v_cos.detach().cpu().numpy(),
        "sigma_log_ratio": sigma_log_ratio.detach().cpu().numpy(),
    }


def phi_subspace(X1, X2, r):
    """φ(i, j) = ‖X1[:, :i]ᵀ X2[:, :j]‖_F² / min(i, j) for i, j ∈ 1..r.

    Invariant to per-column sign flips of ``X1`` / ``X2``.
    """
    M = (X1[:, :r].T @ X2[:, :r]).detach().cpu().numpy()
    M2 = M * M
    cs = np.cumsum(np.cumsum(M2, axis=0), axis=1)
    phi = np.zeros((r, r))
    for i in range(1, r + 1):
        for j in range(1, r + 1):
            phi[i - 1, j - 1] = cs[i - 1, j - 1] / min(i, j)
    return phi


def sigma_spectrum_metrics(S1, S2):
    """Scalar summary of how two singular-value spectra compare."""
    s1 = S1.float()
    s2 = S2.float()
    sigma_cos = (torch.dot(s1, s2) / (s1.norm() * s2.norm() + 1e-30)).item()
    l1 = torch.log1p(s1)
    l2 = torch.log1p(s2)
    sigma_log_cos = (torch.dot(l1, l2) / (l1.norm() * l2.norm() + 1e-30)).item()
    sigma_l1 = (s1 - s2).abs().sum().item()
    return {
        "sigma_cos": sigma_cos,
        "sigma_log_cos": sigma_log_cos,
        "sigma_l1": sigma_l1,
    }


if __name__ == "__main__":
    # Correctness probe: canonicalize reproduces direct SVD of ΔW.
    import sys

    if len(sys.argv) > 1:
        sd = torch.load(sys.argv[1], map_location="cpu")
        a_key = next(k for k in sd if ".lora_A" in k and k.endswith(".weight"))
        mod = a_key.rsplit(".lora_A", 1)[0]
        A = sd[a_key].float()
        B = sd[f"{mod}.lora_B.weight"].float()
    else:
        torch.manual_seed(0)
        A = torch.randn(8, 768)
        B = torch.randn(768, 8)

    r = B.shape[1]
    U, S, V = canonicalize(A, B)
    dW = B @ A
    _, S_ref, _ = torch.linalg.svd(dW, full_matrices=False)
    recon_err = ((U * S) @ V.T - dW).abs().max().item()
    s_err = (S - S_ref[:r]).abs().max().item()
    assert torch.allclose((U * S) @ V.T, dW, atol=1e-4), f"recon err {recon_err}"
    assert torch.allclose(S, S_ref[:r], atol=1e-5), f"S err {s_err}"
    print(f"OK — dW {tuple(dW.shape)}, r={r}, "
          f"recon max err {recon_err:.2e}, S max err {s_err:.2e}")
