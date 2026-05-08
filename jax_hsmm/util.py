"""
Numerical utilities for jax_hsmm.

regularize_for_stability
    Pre-sampling regularization that ensures the MNIW posterior scale matrix
    (Schur complement) is positive definite for every state before IW sampling.

count_frames
    Count total frames across a loaded data dict (list of arrays).

count_frames_wrapper
    Read a PC scores h5 file and print/return total frame count.
"""

import numpy as np
import jax.numpy as jnp


def regularize_for_stability(stats, prior, verbose=True):
    import numpy as np
    import jax.numpy as jnp

    # ----------------------------
    # priors
    # ----------------------------
    M_0   = np.asarray(prior['M_0'], dtype=np.float64)
    V_0   = np.asarray(prior['V_0'], dtype=np.float64)
    Psi_0 = np.asarray(prior['Psi_0'], dtype=np.float64)

    n    = np.asarray(stats['n'], dtype=np.float64)
    S_xx = np.asarray(stats['S_xx'], dtype=np.float64)
    S_xy = np.asarray(stats['S_xy'], dtype=np.float64)
    S_yy = np.asarray(stats['S_yy'], dtype=np.float64)

    K = n.shape[0]
    D_phi = M_0.shape[1]

    I = np.eye(D_phi, dtype=np.float64)

    MV_0 = M_0 @ V_0
    A_prior = Psi_0 + MV_0 @ M_0.T

    # ----------------------------
    # ultra-safe sanitizer
    # ----------------------------
    def clean(A):
        A = np.asarray(A, dtype=np.float64)

        # hard clamp EVERYTHING (this is key)
        A = np.where(np.isfinite(A), A, 0.0)
        A = np.clip(A, -1e10, 1e10)

        if A.shape[0] == A.shape[1]:
          A = 0.5 * (A + A.T)
        return A

    # ----------------------------
    # PSD enforcement WITHOUT eig
    # ----------------------------
    def psd_force(A):
        A = clean(A)

        # diagonal dominance trick (NO eig)
        diag_mean = np.mean(np.diag(A))

        if diag_mean <= 0 or not np.isfinite(diag_mean):
            A = A + (1e-2 + abs(diag_mean)) * np.eye(A.shape[0])

        # ensure strong diagonal stability
        A = A + 1e-6 * np.eye(A.shape[0])

        return A

    # ----------------------------
    # stable Cholesky
    # ----------------------------
    def chol(A):
        A = psd_force(A)
        dim = A.shape[0]

        # Retry with escalating diagonal regularization.
        # Newer NumPy's cholesky can internally call eigvalsh on degenerate
        # matrices, raising "Eigenvalues did not converge" as a LinAlgError.
        # A single fallback isn't enough — we loop until it works.
        for scale in [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]:
            try:
                return np.linalg.cholesky(A + scale * np.eye(dim))
            except (np.linalg.LinAlgError, Exception):
                continue

        # Last resort: return identity so the outer except can reset the state.
        raise np.linalg.LinAlgError("chol failed after all regularization attempts")

    # ----------------------------
    # main loop
    # ----------------------------
    for k in range(K):

        if n[k] < 1:
            continue

        S_xx_k = clean(S_xx[k])
        S_xy_k = clean(S_xy[k])
        S_yy_k = clean(S_yy[k])

        # extra safety: kill pathological scale early
        S_yy_k = np.clip(S_yy_k, -1e8, 1e8)

        B = MV_0 + S_xy_k
        C = V_0 + S_yy_k

        try:
            L = chol(C)
        except Exception:
            if verbose:
                print(f"[WARN] state {k} collapsed → reset")
            S_xx[k] = 0
            S_xy[k] = 0
            S_yy[k] = 0
            n[k] = 0
            continue

        Y = np.linalg.solve(L, B.T)
        Psi_n = A_prior + S_xx_k - Y.T @ Y

        # final stabilization (NO eig)
        Psi_n = clean(Psi_n)
        Psi_n = psd_force(Psi_n)

        S_xx[k] = S_xx_k
        S_xy[k] = S_xy_k
        S_yy[k] = S_yy_k

    return dict(
        n=jnp.array(n),
        S_xx=jnp.array(S_xx),
        S_xy=jnp.array(S_xy),
        S_yy=jnp.array(S_yy),
    )


# ---------------------------------------------------------------------------
# Frame counting
# ---------------------------------------------------------------------------

def count_frames(data_dict: dict) -> int:
    """Count total frames across all sessions in a loaded data dict."""
    return sum(np.asarray(v).shape[0] for v in data_dict.values())


def count_frames_wrapper(
    input_file: str,
    var_name: str = 'scores',
    npcs: int = None,
) -> int:
    """
    Count total frames in a PC scores h5 file and print the result.
    """

    import h5py

    total_frames = 0

    with h5py.File(input_file, 'r') as f:

        if var_name not in f:
            raise KeyError(
                f"Variable '{var_name}' not found in {input_file}. "
                f"Available keys: {list(f.keys())}"
            )

        scores = f[var_name]

        if isinstance(scores, h5py.Group):

            for key in scores:
                total_frames += scores[key].shape[0]

        else:
            shape = scores.shape

            if len(shape) <= 2:
                total_frames = shape[0]
            else:
                total_frames = shape[0] * shape[1]

    print(f'Total frames: {total_frames}')

    return total_frames
