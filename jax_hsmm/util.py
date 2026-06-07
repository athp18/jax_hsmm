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
    """Zero out per-state statistics that contain non-finite values.

    A lightweight pre-pass that resets any state whose sufficient statistics
    contain NaN or Inf to zero counts (forcing a prior draw in sample_obs_params).
    The actual Psi_n PSD enforcement is done inside sample_obs_params via a
    geometric ridge search, avoiding the double Schur-complement computation
    that was in the previous implementation.

    Returns the original stats dict unchanged when no state needed resetting,
    so the caller can detect no-op via ``result is stats``.
    """
    n    = np.asarray(stats['n'],    dtype=np.float64)
    S_xx = np.asarray(stats['S_xx'], dtype=np.float64)
    S_xy = np.asarray(stats['S_xy'], dtype=np.float64)
    S_yy = np.asarray(stats['S_yy'], dtype=np.float64)

    K      = n.shape[0]
    any_reset = False

    for k in range(K):
        if n[k] < 1:
            continue
        if (not np.isfinite(S_xx[k]).all()
                or not np.isfinite(S_xy[k]).all()
                or not np.isfinite(S_yy[k]).all()):
            if verbose:
                print(f"[WARN] state {k} has non-finite stats → reset to prior")
            S_xx[k] = 0.0
            S_xy[k] = 0.0
            S_yy[k] = 0.0
            n[k]    = 0.0
            any_reset = True

    if not any_reset:
        return stats

    return dict(
        n    = jnp.array(n),
        S_xx = jnp.array(S_xx),
        S_xy = jnp.array(S_xy),
        S_yy = jnp.array(S_yy),
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
