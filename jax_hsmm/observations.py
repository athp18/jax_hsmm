"""
Autoregressive Gaussian observation model with Matrix Normal Inverse-Wishart
(MNIW) conjugate prior.

For state k, the generative model is:
    x_t = A_k @ phi_t + eps_t,   eps_t ~ N(0, Sigma_k)

where phi_t = [x_{t-1}; ...; x_{t-lag}] is the stacked lag vector.
If affine=True a constant 1 is appended to phi_t, allowing a bias term.

The MNIW prior (M_0, V_0, Psi_0, nu_0) is conjugate, giving a closed-form
posterior after observing the data assigned to each state.

Log-likelihood computation is jit-compiled and vectorised over all K states
simultaneously.  Parameter sampling uses scipy (fast for small obs_dim).
"""

import numpy as np
from functools import partial
from scipy.stats import invwishart
import jax
import jax.numpy as jnp
from jax import jit


# ---------------------------------------------------------------------------
# Autoregressive feature construction
# ---------------------------------------------------------------------------

def make_ar_features(data: np.ndarray, lag: int, affine: bool = False):
    """Stack lagged observations into a feature matrix.

    Args:
        data:   (T, D) observation array.
        lag:    Number of AR lags.
        affine: If True, append a column of ones to phi (bias term).

    Returns:
        phi: (T - lag, D * lag [+ 1 if affine])  Lagged feature matrix.
        x:   (T - lag, D)                         Corresponding targets.
    """
    T, D = data.shape
    phi = np.hstack([data[lag - i - 1 : T - i - 1] for i in range(lag)])
    x   = data[lag:]
    if affine:
        phi = np.hstack([phi, np.ones((len(phi), 1), dtype=phi.dtype)])
    return phi, x


# ---------------------------------------------------------------------------
# Log-likelihood (jit-compiled)
# ---------------------------------------------------------------------------

@jit
def ar_log_likelihoods(
    phi: jnp.ndarray,
    x: jnp.ndarray,
    A: jnp.ndarray,
    Sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Per-frame per-state log-likelihoods under the AR-Gaussian model.

    Args:
        phi:   (T, D_phi) Lagged feature matrix (may include bias column).
        x:     (T, D)     Target observations.
        A:     (K, D, D_phi) AR weight matrices (last col = bias if affine).
        Sigma: (K, D, D)     Covariance matrices.

    Returns:
        log_liks: (T, K)  log p(x_t | phi_t, state=k, A_k, Sigma_k).
    """
    D = x.shape[1]

    # Predicted means: means[t, k, :] = A[k] @ phi[t]
    means = jnp.einsum('kdi,ti->tkd', A, phi)    # (T, K, D)

    # Residuals
    resid = x[:, None, :] - means                 # (T, K, D)

    # Precision matrices (inverse of Sigma)
    Sigma_inv = jnp.linalg.inv(Sigma)             # (K, D, D)

    # Mahalanobis distances: resid^T Sigma_inv resid
    mah = jnp.einsum('tkd,kde,tke->tk', resid, Sigma_inv, resid)  # (T, K)

    # Log-determinants
    _, log_det = jnp.linalg.slogdet(Sigma)        # (K,)

    log_liks = -0.5 * (mah + log_det[None, :] + D * jnp.log(2.0 * jnp.pi))
    return log_liks                                # (T, K)


# ---------------------------------------------------------------------------
# Student-t log-likelihood (jit-compiled)  — for robust AR model
# ---------------------------------------------------------------------------

@jit
def ar_log_likelihoods_student_t(
    phi: jnp.ndarray,
    x: jnp.ndarray,
    A: jnp.ndarray,
    Sigma: jnp.ndarray,
    nu: float,
) -> jnp.ndarray:
    """Per-frame per-state log-likelihoods under the multivariate Student-t AR model.

    The Student-t arises as the marginal (over precision weights tau) of:
        x_t | phi_t, z_t=k, tau_t ~ N(A_k phi_t, Sigma_k / tau_t)
        tau_t                      ~ Gamma(nu/2, nu/2)

    Closed-form marginal:
        x_t | phi_t, z_t=k ~ t_{nu}(A_k phi_t, Sigma_k)

    Log-PMF:  lgamma((nu+D)/2) - lgamma(nu/2)
              - D/2 * log(nu*pi)
              - 1/2 * log|Sigma|
              - (nu+D)/2 * log(1 + mahal/nu)

    Args:
        phi:   (T, D_phi)    Lagged feature matrix.
        x:     (T, D)        Target observations.
        A:     (K, D, D_phi) AR weight matrices.
        Sigma: (K, D, D)     Scale matrices.
        nu:    float         Degrees of freedom (> 0).

    Returns:
        log_liks: (T, K)
    """
    D = x.shape[1]

    means     = jnp.einsum('kdi,ti->tkd', A, phi)              # (T, K, D)
    resid     = x[:, None, :] - means                          # (T, K, D)
    Sigma_inv = jnp.linalg.inv(Sigma)                          # (K, D, D)
    mah       = jnp.einsum('tkd,kde,tke->tk', resid,
                           Sigma_inv, resid)                    # (T, K)
    _, log_det = jnp.linalg.slogdet(Sigma)                     # (K,)

    log_liks = (
        jax.scipy.special.gammaln((nu + D) / 2.0)
        - jax.scipy.special.gammaln(nu / 2.0)
        - (D / 2.0) * jnp.log(nu * jnp.pi)
        - 0.5 * log_det[None, :]
        - ((nu + D) / 2.0) * jnp.log1p(mah / nu)
    )
    return log_liks                                             # (T, K)


# ---------------------------------------------------------------------------
# Sufficient statistics (jit-compiled)
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(3,))
def compute_sufficient_stats(
    phi: jnp.ndarray,
    x: jnp.ndarray,
    states: jnp.ndarray,
    K: int,
) -> dict:
    """Accumulate MNIW sufficient statistics for every state.

    Args:
        phi:    (T, D_phi) Lagged features (may include bias column).
        x:      (T, D)     Targets.
        states: (T,)       Integer state assignments (0-indexed).
        K:      int        Number of states.

    Returns:
        A dict with keys:
            n    (K,)              Observation counts per state.
            S_xx (K, D, D)         Σ x x^T  per state.
            S_xy (K, D, D_phi)     Σ x phi^T per state.
            S_yy (K, D_phi, D_phi) Σ phi phi^T per state.
    """
    oh = jax.nn.one_hot(states, K)     # (T, K)

    n    = oh.sum(axis=0)                                           # (K,)
    S_xx = jnp.einsum('tk,ti,tj->kij', oh, x,   x  )              # (K, D, D)
    S_xy = jnp.einsum('tk,ti,tj->kij', oh, x,   phi)              # (K, D, D_phi)
    S_yy = jnp.einsum('tk,ti,tj->kij', oh, phi, phi)              # (K, D_phi, D_phi)

    return dict(n=n, S_xx=S_xx, S_xy=S_xy, S_yy=S_yy)


# ---------------------------------------------------------------------------
# Weighted sufficient statistics (for robust AR)  — jit-compiled
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(3,))
def compute_weighted_sufficient_stats(
    phi: jnp.ndarray,
    x: jnp.ndarray,
    states: jnp.ndarray,
    K: int,
    weights: jnp.ndarray,
) -> dict:
    """MNIW sufficient statistics weighted by per-frame precision weights.

    Used for the robust (Student-t) AR model where each frame t is weighted
    by its auxiliary precision weight tau_t ~ Gamma((nu+D)/2, (nu+mahal)/2).
    Multiplying the one-hot assignment matrix by weights[t] is equivalent to
    replacing each x_t x_t^T contribution with tau_t * x_t x_t^T.

    Args:
        phi:     (T, D_phi) Lagged features.
        x:       (T, D)     Targets.
        states:  (T,)       State assignments.
        K:       int        Number of states.
        weights: (T,)       Per-frame precision weights (tau).

    Returns:
        Stats dict (same keys as compute_sufficient_stats).
    """
    oh = jax.nn.one_hot(states, K) * weights[:, None]          # (T, K) weighted

    n    = oh.sum(axis=0)
    S_xx = jnp.einsum('tk,ti,tj->kij', oh, x,   x  )
    S_xy = jnp.einsum('tk,ti,tj->kij', oh, x,   phi)
    S_yy = jnp.einsum('tk,ti,tj->kij', oh, phi, phi)

    return dict(n=n, S_xx=S_xx, S_xy=S_xy, S_yy=S_yy)


@jit
def _mahalanobis_all(
    phi: jnp.ndarray,
    x: jnp.ndarray,
    A: jnp.ndarray,
    Sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Mahalanobis distances for all frames and all states.  (T, K)."""
    means     = jnp.einsum('kdi,ti->tkd', A, phi)              # (T, K, D)
    resid     = x[:, None, :] - means                          # (T, K, D)
    Sigma_inv = jnp.linalg.inv(Sigma)                          # (K, D, D)
    return jnp.einsum('tkd,kde,tke->tk', resid, Sigma_inv, resid)  # (T, K)


def sample_robust_weights(
    rng: np.random.Generator,
    phi: np.ndarray,
    x: np.ndarray,
    states: np.ndarray,
    A: np.ndarray,
    Sigma: np.ndarray,
    nu: float,
) -> np.ndarray:
    """Sample per-frame precision weights tau_t for the robust AR model.

    Given the current state assignment state_t = k, the posterior is:
        tau_t | x_t, phi_t, k  ~  Gamma( (nu+D)/2,  (nu + mahal_k(t))/2 )

    where mahal_k(t) = (x_t - A_k phi_t)^T Sigma_k^{-1} (x_t - A_k phi_t).

    Args:
        rng:    numpy random Generator.
        phi:    (T, D_phi) numpy features.
        x:      (T, D)     numpy targets.
        states: (T,)       integer state assignments.
        A:      (K, D, D_phi) numpy AR matrices.
        Sigma:  (K, D, D)     numpy covariance matrices.
        nu:     float         degrees of freedom.

    Returns:
        tau: (T,) numpy array of sampled precision weights.
    """
    D = x.shape[1]

    # Vectorised Mahalanobis over all (frame, state) pairs.
    mah_all = np.array(
        _mahalanobis_all(
            jnp.array(phi), jnp.array(x),
            jnp.array(A),   jnp.array(Sigma),
        )
    )  # (T, K)

    # Select the distance for the assigned state of each frame.
    state_mah = mah_all[np.arange(len(states)), np.array(states)]  # (T,)

    a = (nu + D) / 2.0
    b = np.maximum((nu + state_mah) / 2.0, 1e-8)
    return rng.gamma(a, 1.0 / b)   # (T,)


# ---------------------------------------------------------------------------
# MNIW posterior sampling (scipy, called once per Gibbs sweep)
# ---------------------------------------------------------------------------

def sample_obs_params(
    stats: dict,
    prior: dict,
    rng: np.random.Generator,
) -> tuple:
    """Sample AR parameters (A, Sigma) for all states via conjugate update.

    Includes a numerical regularization check on Psi_n before IW sampling:
    if the posterior scale matrix is not positive definite (can happen when
    a state has very few observations), a small ridge is added.

    Args:
        stats:  Output of compute_sufficient_stats (JAX or numpy arrays).
        prior:  Dict with keys:
                    M_0   (D, D_phi)       Prior mean of A.
                    V_0   (D_phi, D_phi)   Prior precision on A cols.
                    Psi_0 (D, D)           Prior scale matrix for Sigma.
                    nu_0  float            Prior degrees of freedom.
        rng:    numpy random Generator.

    Returns:
        A_all:     (K, D, D_phi) Sampled AR matrices.
        Sigma_all: (K, D, D)     Sampled covariance matrices.
    """
    M_0, V_0, Psi_0, nu_0 = prior['M_0'], prior['V_0'], prior['Psi_0'], prior['nu_0']
    K     = int(np.array(stats['n']).shape[0])
    D_x   = M_0.shape[0]
    D_phi = M_0.shape[1]

    # Convert JAX arrays to numpy once.
    n    = np.array(stats['n'])
    S_xx = np.array(stats['S_xx'])
    S_xy = np.array(stats['S_xy'])
    S_yy = np.array(stats['S_yy'])

    A_all     = np.zeros((K, D_x, D_phi))
    Sigma_all = np.zeros((K, D_x, D_x))

    for k in range(K):
        n_k = n[k]

        if n_k < D_x + 2:
            # Too few observations: sample directly from prior.
            Sigma_k = invwishart.rvs(df=int(nu_0), scale=Psi_0, random_state=rng)
            eigs, vecs = np.linalg.eigh(Sigma_k)
            if eigs.min() < 1e-6:
                eigs = np.maximum(eigs, 1e-6)
                Sigma_k = vecs @ np.diag(eigs) @ vecs.T
                Sigma_k = (Sigma_k + Sigma_k.T) * 0.5
            L_S = np.linalg.cholesky(Sigma_k)
            L_V = np.linalg.cholesky(np.linalg.inv(V_0))
            Z   = rng.standard_normal((D_x, D_phi))
            A_k = M_0 + L_S @ Z @ L_V.T
        else:
            # Posterior MNIW update.
            V_n = np.array(V_0 + S_yy[k], dtype=np.float64)
            # Symmetrise — S_yy[k] comes from JAX float32 and can introduce
            # tiny asymmetries.
            V_n = (V_n + V_n.T) * 0.5

            # Robust Cholesky: escalate ridge geometrically until V_n is PD.
            # Starting ridge is relative to the diagonal scale so it adapts
            # to matrices with large (many-frame) or small entries.
            diag_scale = max(float(np.abs(np.diag(V_n)).mean()), 1.0)
            lam_v = 0.0
            V_n_chol = None
            for _v_iter in range(200):
                try:
                    V_n_chol = np.linalg.cholesky(V_n + lam_v * np.eye(D_phi))
                    if lam_v > 0.0:
                        V_n = V_n + lam_v * np.eye(D_phi)
                    break
                except np.linalg.LinAlgError:
                    lam_v = max(lam_v * 2.0, diag_scale * 1e-8)
            if V_n_chol is None:
                # Extreme fallback: use prior precision only.
                V_n = np.array(V_0, dtype=np.float64)
                V_n_chol = np.linalg.cholesky(V_n)

            M_n = np.linalg.solve(V_n, (S_xy[k] + M_0 @ V_0).T).T

            nu_n  = nu_0 + n_k
            Psi_n = (Psi_0
                     + S_xx[k]
                     + M_0 @ V_0 @ M_0.T
                     - M_n @ V_n @ M_n.T)
            # Symmetrise for numerical stability.
            Psi_n = (Psi_n + Psi_n.T) * 0.5

            # If Psi_n contains non-finite values (NaN/Inf from exploding
            # sufficient statistics that slipped past regularize_for_stability),
            # fall back to a prior draw for this state.
            if not np.isfinite(Psi_n).all():
                Sigma_k = invwishart.rvs(df=int(nu_0), scale=Psi_0, random_state=rng)
                L_S = np.linalg.cholesky(Sigma_k)
                L_V = np.linalg.cholesky(np.linalg.inv(V_0))
                Z   = rng.standard_normal((D_x, D_phi))
                A_k = M_0 + L_S @ Z @ L_V.T
                A_all[k]     = A_k
                Sigma_all[k] = Sigma_k
                continue

            # Regularisation safety check: Psi_n must be PD for IW sampling.
            min_eig = np.linalg.eigvalsh(Psi_n).min()
            if min_eig < 1e-8:
                Psi_n += (1e-6 - min_eig + 0.01 * np.abs(np.diag(Psi_n)).mean()) * np.eye(D_x)

            # Sample Sigma_k ~ IW(Psi_n, nu_n).
            Sigma_k = invwishart.rvs(df=int(nu_n), scale=Psi_n, random_state=rng)

            # Sample A_k | Sigma_k ~ MN(M_n, Sigma_k, V_n^{-1}).
            # V_n = L_V L_V^T  →  sample = M_n + L_S @ Z @ L_V^{-1}
            # Avoid cholesky(V_n^{-1}) which loses PD under float32 rounding.
            L_S   = np.linalg.cholesky(Sigma_k)
            L_V_i = np.linalg.solve(V_n_chol, np.eye(D_phi))  # L_V^{-1}
            Z     = rng.standard_normal((D_x, D_phi))
            A_k   = M_n + L_S @ Z @ L_V_i

        A_all[k]     = A_k
        Sigma_all[k] = Sigma_k

    return jnp.array(A_all), jnp.array(Sigma_all)


# ---------------------------------------------------------------------------
# Prior construction helpers
# ---------------------------------------------------------------------------

def default_mniw_prior(obs_dim: int, ar_lags: int, affine: bool = False) -> dict:
    """Weakly informative MNIW prior centred on a lag-1 identity AR matrix.

    The prior mean M_0 puts an identity on the first obs_dim x obs_dim block
    (lag-1 persistence) and zeros elsewhere — a sensible default for
    smoothly-varying PCA trajectories.

    Args:
        obs_dim: Observation dimensionality D.
        ar_lags: Number of AR lags.
        affine:  If True, phi has an extra bias column (D_phi = D*lag + 1).

    Returns:
        prior dict compatible with sample_obs_params.
    """
    D      = obs_dim
    D_phi  = D * ar_lags + (1 if affine else 0)

    M_0         = np.zeros((D, D_phi))
    M_0[:D, :D] = np.eye(D)     # lag-1 identity block

    return dict(
        M_0   = M_0,
        V_0   = 10.0 * np.eye(D_phi),   # matches moseq2-model K_0_scale=10.0
        Psi_0 = np.eye(D) * 0.01,        # matches moseq2-model S_0_scale=0.01
                                          # E[Sigma]=Psi_0/(nu_0-D-1)=0.01*I; keeps
                                          # prior draws tight so low-count states
                                          # don't steal frames via inflated likelihoods
        nu_0  = float(D + 2),
    )


def empirical_bayes_mniw_prior(
    data_list: list,
    obs_dim: int,
    ar_lags: int,
    affine: bool = False,
    nu_0: float = None,
) -> dict:
    """Data-informed MNIW prior via global AR maximum-likelihood fit.

    Fits a single AR model to all data (pooled) and uses the MLE residual
    covariance to set Psi_0, matching moseq2-model's empirical Bayes init.

        Psi_0 = Sigma_mle * (nu_0 - D - 1)   so that  E[Sigma] = Sigma_mle.

    The prior mean M_0 retains the lag-1 identity structure (same as the
    default prior) — moseq2-model only borrows the covariance estimate, not
    the AR coefficient estimate, for the prior mean.

    Args:
        data_list: List of (T_i, D) float arrays (PCA scores, pre-whitened).
        obs_dim:   Observation dimensionality D.
        ar_lags:   Number of AR lags.
        affine:    If True, phi has an extra bias column.
        nu_0:      IW prior degrees of freedom (default: D + 2).

    Returns:
        prior dict compatible with sample_obs_params.
    """
    D     = obs_dim
    D_phi = D * ar_lags + (1 if affine else 0)

    if nu_0 is None:
        nu_0 = float(D + 2)

    # Accumulate global sufficient statistics in float64 for numerical safety.
    S_yy    = np.zeros((D_phi, D_phi), dtype=np.float64)
    S_xy    = np.zeros((D, D_phi),     dtype=np.float64)
    S_xx    = np.zeros((D, D),         dtype=np.float64)
    n_total = 0

    for data in data_list:
        phi, x = make_ar_features(
            np.asarray(data, dtype=np.float64), ar_lags, affine=affine
        )
        S_yy    += phi.T @ phi
        S_xy    += x.T   @ phi
        S_xx    += x.T   @ x
        n_total += len(x)

    # Fall back to default prior if not enough data to fit.
    if n_total < D_phi + D + 2:
        return default_mniw_prior(obs_dim, ar_lags, affine=affine)

    # Global MLE: solve S_yy A_mle^T = S_xy^T (avoids explicit inverse).
    try:
        A_mle = np.linalg.solve(S_yy, S_xy.T).T          # (D, D_phi)
    except np.linalg.LinAlgError:
        return default_mniw_prior(obs_dim, ar_lags, affine=affine)

    # MLE residual covariance  Sigma_mle = (Sxx - A_mle Syy A_mle^T) / n
    Sigma_mle = (S_xx - A_mle @ S_yy @ A_mle.T) / n_total
    Sigma_mle = (Sigma_mle + Sigma_mle.T) * 0.5       # symmetrise

    # Ensure positive definiteness.
    min_eig = np.linalg.eigvalsh(Sigma_mle).min()
    if min_eig < 1e-6:
        Sigma_mle += (1e-6 - min_eig + 1e-8) * np.eye(D)

    # Set Psi_0 so that E[Sigma] = Sigma_mle under IW(Psi_0, nu_0).
    # E[IW(Psi_0, nu_0)] = Psi_0 / (nu_0 - D - 1)  → Psi_0 = Sigma_mle * df
    df = nu_0 - D - 1.0
    if df <= 0.0:
        df = 1.0
    Psi_0 = Sigma_mle * df

    # M_0: lag-1 identity prior (same as default_mniw_prior).
    M_0         = np.zeros((D, D_phi), dtype=np.float64)
    M_0[:D, :D] = np.eye(D)

    return dict(
        M_0   = M_0,
        V_0   = 10.0 * np.eye(D_phi, dtype=np.float64),   # matches moseq2-model K_0_scale=10.0
        Psi_0 = Psi_0,
        nu_0  = nu_0,
    )
