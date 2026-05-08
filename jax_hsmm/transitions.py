"""
Sticky HDP-HMM transition distribution — Gibbs updates.

Implements the auxiliary-variable sampler from:
    Fox, Sudderth, Jordan, Willsky (2008)
    "An HDP-HMM for Systems with State Persistence"

The sticky HDP-HMM adds a self-transition bias (kappa) to the HDP prior,
which encourages the model to stay in the same state longer — important for
segmenting behavior into syllables with realistic durations.

Generative model
----------------
    beta ~ GEM(gamma)                            [global mixture weights]
    pi_k ~ Dir(alpha * beta + kappa * e_k)       [row-specific transitions]
    z_t  | z_{t-1}=k ~ Categorical(pi_k)

Gibbs update
------------
1. Count transitions n[i, j] from the current state sequence.
2. Sample auxiliary CRF table counts m[i, j] from the CRP posterior.
3. Sample beta  ~ Dir(m_bar[.] + gamma / K)
4. Sample pi[k] ~ Dir(alpha * beta + kappa * e_k + n[k, :])

All operations are numpy/scipy (cheap compared to message passing).
"""

import numpy as np
from scipy.special import digamma, polygamma


# ---------------------------------------------------------------------------
# Transition-count accumulation
# ---------------------------------------------------------------------------

# transitions.py — replace lines 47-49:
def count_transitions(states: np.ndarray, K: int) -> np.ndarray:
    counts = np.zeros((K, K), dtype=np.int32)
    s = np.asarray(states)
    np.add.at(counts, (s[:-1], s[1:]), 1)
    return counts


def count_transitions_batch(state_seqs: list, K: int) -> np.ndarray:
    """Sum transition counts over multiple sessions.

    Args:
        state_seqs: List of (T_i,) integer state sequences.
        K:          Number of states.

    Returns:
        counts: (K, K) cumulative transition count matrix.
    """
    counts = np.zeros((K, K), dtype=np.int32)
    for seq in state_seqs:
        counts += count_transitions(np.asarray(seq), K)
    return counts


# ---------------------------------------------------------------------------
# CRF auxiliary variable sampling (table counts)
# ---------------------------------------------------------------------------

def _sample_crp_tables(rng: np.random.Generator, n: int, concentration: float) -> int:
    if n == 0 or concentration <= 0.0:
        return 0
    
    if n <= 200:
        # Exact sequential sampler for small n
        m = 0
        for seat in range(1, n + 1):
            m += int(rng.random() < concentration / (concentration + seat - 1))
        return m
    
    # For large n: use the exact mean and variance of m
    # under the Chinese Restaurant Process.
    #
    # E[m]   = concentration * (psi(concentration + n) - psi(concentration))
    # Var[m] = concentration * (psi_1(concentration) - psi_1(concentration + n))
    #
    # where psi_1 is the trigamma function.
    # Both are exact results from the Stirling number generating function.
    # We approximate m ~ Normal(mean, var) rounded to nearest int,
    # which is accurate when both mean and var are large
    
    mean = concentration * (
        digamma(concentration + n) - digamma(concentration)
    )
    var = concentration * (
        polygamma(1, concentration) - polygamma(1, concentration + n)
    )

    # polygamma(1, x) > 0 always, but floating-point cancellation can produce
    # var ≈ 0 when n is very large (both terms nearly equal).  Fall back to the
    # rounded mean when variance is too small for a useful Normal approximation.
    if var < 1e-10 or mean <= 0:
        return int(np.clip(round(mean), 1, n))

    # Normal approximation — accurate when mean >> 1
    sample = rng.normal(mean, np.sqrt(var))
    return int(np.clip(round(sample), 1, n))


def sample_crp_table_counts(
    rng: np.random.Generator,
    n: np.ndarray,
    alpha: float,
    kappa: float,
    beta: np.ndarray,
) -> np.ndarray:
    """Sample auxiliary CRF table counts m[i, j] for all state pairs.

    Args:
        rng:   numpy random Generator.
        n:     (K, K) transition counts.
        alpha: HDP concentration parameter.
        kappa: Stickiness parameter (extra weight on self-transitions).
        beta:  (K,) global mixture weights.

    Returns:
        m: (K, K) table count matrix.
    """
    K = n.shape[0]
    m = np.zeros((K, K), dtype=np.int32)
    for i in range(K):
        for j in range(K):
            if n[i, j] == 0:
                continue  # _sample_crp_tables returns 0, skip the call
            conc = alpha * beta[j] + (kappa if i == j else 0.0)
            m[i, j] = _sample_crp_tables(rng, int(n[i, j]), conc)
    return m


# ---------------------------------------------------------------------------
# Beta and pi sampling
# ---------------------------------------------------------------------------

def sample_beta(
    rng: np.random.Generator,
    m: np.ndarray,
    gamma: float,
    alpha: float = 0.0,
    kappa: float = 0.0,
    beta: np.ndarray = None,
) -> np.ndarray:
    K = m.shape[0]
    m_corrected = m.copy().astype(float)

    # Fox et al (2008) override correction for sticky HDP-HMM.
    # Thin each diagonal entry: keep only the tables that represent genuine
    # global dish k popularity, stripping out the kappa self-transition bonus.
    # keep_prob = alpha*beta[k] / (alpha*beta[k] + kappa)
    if kappa > 0.0 and beta is not None:
        diag_idx = np.arange(K)
        diag_vals = m[diag_idx, diag_idx].astype(int)
        nonzero = diag_vals > 0
        if nonzero.any():
            denom = alpha * np.asarray(beta) + kappa
            keep_prob = np.where(denom > 0, alpha * np.asarray(beta) / denom, 0.0)
            keep_prob = np.clip(keep_prob, 0.0, 1.0)
            thinned = rng.binomial(diag_vals[nonzero], keep_prob[nonzero])
            m_corrected[diag_idx[nonzero], diag_idx[nonzero]] = thinned

    m_bar = m_corrected.sum(axis=0)
    concentration = np.maximum(m_bar + gamma / K, 1e-8)
    return rng.dirichlet(concentration)


def sample_pi(
    rng: np.random.Generator,
    n: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    kappa: float,
) -> np.ndarray:
    """Sample each row of the transition matrix pi_k ~ Dir(alpha*beta + kappa*e_k + n_k).

    Args:
        rng:   numpy random Generator.
        n:     (K, K) transition counts.
        beta:  (K,) global mixture weights.
        alpha: HDP concentration.
        kappa: Stickiness parameter.

    Returns:
        pi: (K, K) transition matrix (rows sum to 1).
    """
    K = n.shape[0]
    pi = np.zeros((K, K))
    for k in range(K):
        concentration = alpha * beta + n[k].astype(float)
        concentration[k] += kappa
        concentration = np.maximum(concentration, 1e-8)
        pi[k] = rng.dirichlet(concentration)
    return pi


# ---------------------------------------------------------------------------
# Full HDP-HMM transition Gibbs step
# ---------------------------------------------------------------------------

def sample_transitions(
    rng: np.random.Generator,
    state_seqs: list,
    beta: np.ndarray,
    params: dict,
) -> tuple:
    """One complete Gibbs update of (beta, pi).

    Args:
        rng:       numpy random Generator.
        state_seqs: List of (T_i,) integer state sequences.
        beta:      (K,) current global mixture weights.
        params:    Dict with keys: K, alpha, kappa, gamma.

    Returns:
        (pi_new, beta_new):
            pi_new:   (K, K) updated transition matrix.
            beta_new: (K,)   updated global weights.
    """
    K     = params['K']
    alpha = params['alpha']
    kappa = params['kappa']
    gamma = params['gamma']

    # 1. Accumulate transition counts across all sessions.
    n = count_transitions_batch(state_seqs, K)

    # 2. Sample auxiliary table counts.
    m = sample_crp_table_counts(rng, n, alpha, kappa, beta)

    # 3. Update global weights.
    beta_new = sample_beta(rng, m, gamma)

    # 4. Update per-row transition distributions.
    pi_new = sample_pi(rng, n, beta_new, alpha, kappa)

    return pi_new, beta_new


# ---------------------------------------------------------------------------
# Separate-transition sampler (one pi_g per experimental group)
# ---------------------------------------------------------------------------

def sample_transitions_separate(
    rng: np.random.Generator,
    state_seqs_by_group: dict,
    beta: np.ndarray,
    params: dict,
) -> tuple:
    """Sample a separate transition matrix for each group while sharing beta.

    Each group g has its own per-row counts n_g[i,j] and gets a distinct
    pi_g[k] ~ Dir(alpha*beta + kappa*e_k + n_g[k,:]).  Beta is updated from
    the pooled CRF table counts across all groups, preserving the HDP prior
    coupling between groups.

    Mirrors moseq2-model's FastARWeakLimitStickyHDPHMMSeparateTrans.

    Args:
        rng:                 numpy random Generator.
        state_seqs_by_group: Dict {group_id: list of (T_i,) state sequences}.
        beta:                (K,) current global mixture weights.
        params:              Dict with keys: K, alpha, kappa, gamma.

    Returns:
        (pi_groups, beta_new):
            pi_groups: dict {group_id: (K, K)} per-group transition matrices.
            beta_new:  (K,) updated global weights.
    """
    K     = params['K']
    alpha = params['alpha']
    kappa = params['kappa']
    gamma = params['gamma']

    n_per_group = {}
    all_m = np.zeros((K, K), dtype=np.int32)

    for g, seqs in state_seqs_by_group.items():
        n_g = count_transitions_batch(seqs, K)
        n_per_group[g] = n_g
        m_g = sample_crp_table_counts(rng, n_g, alpha, kappa, beta)
        all_m += m_g

    # Update shared beta from pooled table counts.
    beta_new = sample_beta(rng, all_m, gamma)

    # Update per-group pi.
    pi_groups = {
        g: sample_pi(rng, n_g, beta_new, alpha, kappa)
        for g, n_g in n_per_group.items()
    }

    return pi_groups, beta_new


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def init_transitions(K: int, alpha: float, kappa: float, gamma: float) -> dict:
    """Return a params dict and uniform initial (pi, beta).

    Args:
        K:     Number of states.
        alpha: HDP row concentration.
        kappa: Stickiness bonus.
        gamma: Top-level concentration.

    Returns:
        params: Dict of hyperparameters.
        pi:     (K, K) uniform transition matrix.
        beta:   (K,)   uniform global weights.
    """
    params = dict(K=K, alpha=alpha, kappa=kappa, gamma=gamma)
    pi     = np.full((K, K), 1.0 / K)
    beta   = np.full((K,),   1.0 / K)
    return params, pi, beta
