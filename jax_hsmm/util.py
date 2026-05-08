"""
Numerical utilities for jax_hsmm.

regularize_for_stability
    Pre-sampling regularization that ensures the MNIW posterior scale matrix
    (Schur complement) is positive definite for every state before IW sampling.

    Mirrors the role of moseq2-model's `regularize_for_stability`, but
    operates on plain numpy stats dicts rather than `pyhsmm` obs_distns objects.
"""

import numpy as np
import jax.numpy as jnp


def regularize_for_stability(
    stats: dict,
    prior: dict,
    verbose: bool = False,
) -> dict:
    """Regularize sufficient statistics so the MNIW posterior is well-conditioned.

    For each state k with enough observations (n_k >= D_x + 2), we check that
    the posterior scale matrix

        Psi_n(k) = Psi_0 + M_0 V_0 M_0' + S_xx[k]
                   - (M_0 V_0 + S_xy[k]) (V_0 + S_yy[k])^{-1} (M_0 V_0 + S_xy[k])'

    has strictly positive minimum eigenvalue.  If not, we add a ridge
    ``lambda * I`` to ``S_yy[k]``, which increases ``C = V_0 + S_yy[k]`` and
    moves Psi_n(k) toward PD.  Lambda is grown geometrically (factor 1.05)
    starting from 1e-4 until the Schur complement is PD.

    This matches the behaviour of moseq2-model/train/util.py
    ``regularize_for_stability``, which calls
    ``obs_distns[k].regularize_for_stability(kappa)`` — the kappa there is the
    ridge being added to the gram matrix (equivalent to S_yy here).

    Called in ARHMM._gibbs_step **after** sufficient statistics are accumulated
    and **before** sample_obs_params, gated on ``self._obs_stats is not None``
    (i.e. skipped on the very first sweep where stats come from dummy data).

    Args:
        stats:   Stats dict (keys: n, S_xx, S_xy, S_yy) as returned by
                 compute_sufficient_stats or compute_weighted_sufficient_stats.
                 JAX arrays are converted to numpy internally; the returned
                 dict has jnp arrays for S_yy if any state was modified.
        prior:   MNIW prior dict (keys: M_0, V_0, Psi_0, nu_0).
        verbose: If True, print which states were regularized and by how much.

    Returns:
        Stats dict.  If no state needed regularization the *same* dict object
        is returned unchanged.  Otherwise a new dict is returned with S_yy
        replaced by a jnp array that incorporates the ridge corrections.
    """
    M_0  = np.asarray(prior['M_0'],  dtype=np.float64)
    V_0  = np.asarray(prior['V_0'],  dtype=np.float64)
    Psi_0 = np.asarray(prior['Psi_0'], dtype=np.float64)

    n    = np.array(stats['n'],    dtype=np.float64)
    S_xx = np.array(stats['S_xx'], dtype=np.float64)   # (K, D_x, D_x)
    S_xy = np.array(stats['S_xy'], dtype=np.float64)   # (K, D_x, D_phi)
    S_yy = np.array(stats['S_yy'], dtype=np.float64)   # (K, D_phi, D_phi)

    K     = n.shape[0]
    D_x   = M_0.shape[0]
    D_phi = M_0.shape[1]
    I_phi = np.eye(D_phi, dtype=np.float64)

    # Precompute prior contribution (state-independent).
    MV_0    = M_0 @ V_0                # (D_x, D_phi)
    A_prior = Psi_0 + MV_0 @ M_0.T    # (D_x, D_x)

    any_modified = False

    for k in range(K):
        if n[k] < D_x + 2:
            # Fewer observations than needed for a proper posterior:
            # sample_obs_params will draw from the prior, so skip.
            continue

        B = MV_0 + S_xy[k]   # (D_x, D_phi)

        lam = 0.0
        is_pd = False

        for _iter in range(300):        # safety cap on iterations
            C = V_0 + S_yy[k] + lam * I_phi    # (D_phi, D_phi)
            try:
                C_chol = np.linalg.cholesky(C)
                # Schur: A_prior + S_xx[k] - B C^{-1} B'
                # Use cholesky solve for numerical safety: C^{-1} B' = (C \ B')
                C_inv_Bt = np.linalg.solve(C, B.T)   # (D_phi, D_x)
                Psi_n = A_prior + S_xx[k] - B @ C_inv_Bt
            except np.linalg.LinAlgError:
                lam = max(lam * 1.1, 1e-6)
                continue

            Psi_n = (Psi_n + Psi_n.T) * 0.5       # symmetrise
            min_eig = np.linalg.eigvalsh(Psi_n).min()

            if min_eig > 1e-8:
                is_pd = True
                break

            # Escalate ridge geometrically.
            lam = max(lam * 1.05, 1e-4)

        if lam > 0.0:
            if verbose:
                print(
                    f"regularize_for_stability: state {k:3d}  "
                    f"n={int(n[k])}  ridge λ={lam:.3e}"
                )
            S_yy[k] += lam * I_phi
            any_modified = True

    if any_modified:
        return dict(
            n    = stats['n'],
            S_xx = stats['S_xx'],
            S_xy = stats['S_xy'],
            S_yy = jnp.array(S_yy),
        )
    return stats
