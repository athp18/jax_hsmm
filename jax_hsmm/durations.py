"""
Duration distributions for the HSMM.

Each state k has an independent duration distribution p_k(d) over positive
integers.  We use a Poisson distribution with a Gamma conjugate prior:

    d | lambda_k ~ Poisson(lambda_k)      (truncated at max_dur)
    lambda_k     ~ Gamma(a_0, b_0)

The posterior is Gamma(a_0 + sum_d, b_0 + n_segs) where the sum runs over
all observed segment durations assigned to state k and n_segs is the number
of such segments.

This is the simplest conjugate choice.  The Negative-Binomial with a
Beta-prime prior is more flexible (heavier tails) but requires more
hyperparameter tuning; it can be substituted by implementing
`log_dur_matrix` and `sample_duration_params` below.
"""

import numpy as np
from functools import partial
from scipy.stats import gamma as gamma_dist
import jax.numpy as jnp
from jax import jit


# ---------------------------------------------------------------------------
# Log-duration probability matrix (jit-compiled)
# ---------------------------------------------------------------------------

@partial(jit, static_argnums=(1,))
def poisson_log_dur_matrix(
    lam: jnp.ndarray,
    max_dur: int,
) -> jnp.ndarray:
    """Compute a (K, max_dur) matrix of truncated Poisson log-PMF values.

    log_dur[k, d-1] = log P(dur = d | state = k), d = 1 .. max_dur.
    The PMF is truncated to [1, max_dur] and renormalised.

    Args:
        lam:     (K,) Poisson rate parameters (one per state).
        max_dur: int  Maximum duration.

    Returns:
        log_dur: (K, max_dur)
    """
    d = jnp.arange(1, max_dur + 1, dtype=jnp.float32)   # (max_dur,)

    # log(d!) = cumsum(log(1), log(2), ..., log(d)) computed entirely in JAX
    # so it is part of the compiled graph and avoids a Python loop at trace time.
    log_factorial = jnp.concatenate(
        [jnp.zeros(1), jnp.cumsum(jnp.log(jnp.arange(1, max_dur + 1, dtype=jnp.float32)))]
    )[1:]  # shape (max_dur,); entry i = log((i+1)!)

    # log Poisson PMF (unnormalised over d >= 1):
    # log p(d | lambda) = d * log(lambda) - lambda - log(d!)
    log_pmf = (d[None, :] * jnp.log(lam[:, None])
               - lam[:, None]
               - log_factorial[None, :])
    # (K, max_dur)

    # Normalise over the truncated support [1, max_dur].
    log_normaliser = jnp.log(
        jnp.sum(jnp.exp(log_pmf - log_pmf.max(axis=1, keepdims=True)), axis=1,
                keepdims=True)
    ) + log_pmf.max(axis=1, keepdims=True)
    return log_pmf - log_normaliser


def _log_factorial(n: int) -> float:
    """log(n!) — retained for any external callers; no longer used internally."""
    if n <= 1:
        return 0.0
    val = 0.0
    for i in range(2, n + 1):
        val += np.log(i)
    return val


# ---------------------------------------------------------------------------
# Segment information extraction
# ---------------------------------------------------------------------------

def get_segment_durations(states: np.ndarray) -> tuple:
    """Extract per-segment (state, duration) pairs from a state sequence.

    Args:
        states: (T,) integer state assignments.

    Returns:
        seg_states:    (n_segs,) state label for each segment.
        seg_durations: (n_segs,) duration (frame count) for each segment.
    """
    seg_states    = []
    seg_durations = []
    if len(states) == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    current_state = states[0]
    count = 1
    for t in range(1, len(states)):
        if states[t] == current_state:
            count += 1
        else:
            seg_states.append(current_state)
            seg_durations.append(count)
            current_state = states[t]
            count = 1
    seg_states.append(current_state)
    seg_durations.append(count)

    return np.array(seg_states, dtype=np.int32), np.array(seg_durations, dtype=np.int32)


def accumulate_duration_stats(state_seqs: list, K: int) -> tuple:
    """Accumulate sufficient statistics for duration parameters across sessions.

    For the Poisson-Gamma model the sufficient statistics per state k are:
        sum_dur_k  = Σ_{segments with state k} d_n   (total frames)
        n_segs_k   = number of segments in state k

    Args:
        state_seqs: List of (T_i,) integer state sequences (numpy arrays).
        K:          Number of states.

    Returns:
        sum_dur:  (K,) total duration frames per state.
        n_segs:   (K,) number of segments per state.
    """
    sum_dur = np.zeros(K, dtype=np.int64)
    n_segs  = np.zeros(K, dtype=np.int64)
    for seq in state_seqs:
        s_states, s_durs = get_segment_durations(np.asarray(seq))
        for k in range(K):
            mask = (s_states == k)
            n_segs[k]  += mask.sum()
            sum_dur[k] += s_durs[mask].sum()
    return sum_dur, n_segs


# ---------------------------------------------------------------------------
# Duration parameter Gibbs update
# ---------------------------------------------------------------------------

def sample_duration_params(
    rng: np.random.Generator,
    state_seqs: list,
    lam: np.ndarray,
    prior: dict,
    K: int,
) -> np.ndarray:
    """Sample Poisson rate parameters lambda_k via Gamma-Poisson conjugate update.

    Posterior:
        lambda_k | data ~ Gamma(a_0 + sum_dur_k, b_0 + n_segs_k)

    Args:
        rng:       numpy random Generator.
        state_seqs: List of (T_i,) integer state sequences.
        lam:       (K,) current rate parameters (used if n_segs_k == 0).
        prior:     Dict with keys: a_0 (float), b_0 (float).
        K:         Number of states.

    Returns:
        lam_new: (K,) updated Poisson rate parameters.
    """
    a_0 = prior['a_0']
    b_0 = prior['b_0']

    sum_dur, n_segs = accumulate_duration_stats(state_seqs, K)

    lam_new = np.empty(K)
    for k in range(K):
        a_n = a_0 + float(sum_dur[k])
        b_n = b_0 + float(n_segs[k])
        if n_segs[k] == 0:
            # No observations: sample from prior.
            lam_new[k] = rng.gamma(a_0, 1.0 / b_0)
        else:
            # Sample from Gamma(a_n, 1/b_n)  [shape, rate parameterisation].
            lam_new[k] = rng.gamma(a_n, 1.0 / b_n)

    return lam_new


# ---------------------------------------------------------------------------
# Prior construction helper
# ---------------------------------------------------------------------------

def default_duration_prior(expected_dur: float = 20.0) -> dict:
    """Weakly informative Gamma prior centred on expected_dur.

    Sets a_0 = 2, b_0 = 2 / expected_dur so that E[lambda] = expected_dur.

    Args:
        expected_dur: Prior expected duration in frames.

    Returns:
        Dict with a_0 and b_0.
    """
    a_0 = 2.0
    b_0 = a_0 / expected_dur
    return dict(a_0=a_0, b_0=b_0)
