"""
JAX implementations of HMM and HSMM forward-backward message passing.

Replaces the Cython hot-loops in the original pyhsmm
(hmm_messages_interface.pyx, hsmm_messages_interface.pyx).

All performance-critical paths are jit-compiled and use jax.lax.scan
in place of Python for-loops, making them GPU-compatible with no code
changes.  The HSMM backward sampler uses a Python while-loop because
the number of segments is data-dependent and cannot be statically shaped.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial


# ---------------------------------------------------------------------------
# HMM message passing
# ---------------------------------------------------------------------------

@jit
def hmm_forward(
    log_pi0: jnp.ndarray,
    log_A: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> jnp.ndarray:
    """Compute HMM forward (alpha) messages in log space.

    Args:
        log_pi0:         (K,)   Log initial-state probabilities.
        log_A:           (K, K) Log transition matrix.
                                log_A[i, j] = log P(z_t=j | z_{t-1}=i).
        log_likelihoods: (T, K) Log observation likelihoods per state.

    Returns:
        log_alphas: (T, K)  log_alphas[t, k] ≈ log P(x_{1:t+1}, z_t=k).
    """
    def step(log_alpha, log_lik_t):
        # Predict: sum over from-states along axis 0 of log_A
        log_alpha_new = log_lik_t + jax.nn.logsumexp(
            log_alpha[:, None] + log_A, axis=0   # (K, K) -> (K,)
        )
        return log_alpha_new, log_alpha_new

    log_alpha_0 = log_pi0 + log_likelihoods[0]
    _, log_alphas_rest = lax.scan(step, log_alpha_0, log_likelihoods[1:])
    return jnp.concatenate([log_alpha_0[None], log_alphas_rest], axis=0)


@jit
def hmm_backward_sample(
    key: jnp.ndarray,
    log_A: jnp.ndarray,
    log_alphas: jnp.ndarray,
) -> jnp.ndarray:
    """Sample a state sequence via forward-filtering backward-sampling (FFBS).

    Args:
        key:        JAX PRNGKey.
        log_A:      (K, K) Log transition matrix.
        log_alphas: (T, K) Forward messages from hmm_forward().

    Returns:
        states: (T,) Sampled integer state sequence.
    """
    # Sample z_{T-1} from the marginal given all observations.
    key, subkey = jax.random.split(key)
    z_last = jax.random.categorical(subkey, log_alphas[-1])

    def step(carry, log_alpha_t):
        key, z_next = carry
        key, subkey = jax.random.split(key)
        # P(z_t | z_{t+1}, x_{1:t}) ∝ alpha_t(z_t) * A[z_t, z_{t+1}]
        log_probs = log_alpha_t + log_A[:, z_next]
        z_t = jax.random.categorical(subkey, log_probs)
        return (key, z_t), z_t

    # Scan backwards through time steps 0 .. T-2 (reversed).
    (_, _), z_rest = lax.scan(
        step, (key, z_last), log_alphas[:-1][::-1]
    )

    # z_rest = [z_{T-2}, ..., z_0]; reverse and append z_{T-1}.
    return jnp.concatenate([z_rest[::-1], jnp.array([z_last])])


@jit
def hmm_backward_msgs(
    log_A: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> jnp.ndarray:
    """Compute HMM backward (beta) messages in log space.

    Args:
        log_A:           (K, K) Log transition matrix.
                                log_A[i, j] = log P(z_t=j | z_{t-1}=i).
        log_likelihoods: (T, K) Log observation likelihoods per state.

    Returns:
        log_betas: (T, K)  log_betas[t, k] ≈ log P(x_{t+2:T} | z_t=k).
                           log_betas[T-1, :] = 0 by convention.
    """
    log_beta_T = jnp.zeros(log_likelihoods.shape[1])   # (K,)

    def step(log_beta_next, log_lik_t1):
        # log_beta[t, k] = logsumexp_j ( log_A[k, j] + log_lik[t+1, j] + log_beta[t+1, j] )
        log_beta_t = jax.nn.logsumexp(
            log_A + log_lik_t1[None, :] + log_beta_next[None, :],   # (K, K)
            axis=1,   # sum over j  ->  (K,)
        )
        return log_beta_t, log_beta_t

    # Scan backwards: xs = log_likelihoods[1:][::-1]  (shape T-1, K)
    # Output ys[i] = log_beta[T-2-i], so ys[::-1] = [log_beta[0], ..., log_beta[T-2]]
    _, log_betas_rev = lax.scan(step, log_beta_T, log_likelihoods[1:][::-1])

    return jnp.concatenate([log_betas_rev[::-1], log_beta_T[None]], axis=0)


@jit
def hmm_expected_states(
    log_pi0: jnp.ndarray,
    log_A: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> jnp.ndarray:
    """Compute posterior state marginals p(z_t=k | x_{1:T}) via forward-backward.

    Implements the E-step of the Baum-Welch / forward-backward algorithm.

    Args:
        log_pi0:         (K,)   Log initial-state probabilities.
        log_A:           (K, K) Log transition matrix.
        log_likelihoods: (T, K) Log observation likelihoods per state.

    Returns:
        expected_states: (T, K)  Posterior state marginals (rows sum to 1).
    """
    log_alphas = hmm_forward(log_pi0, log_A, log_likelihoods)      # (T, K)
    log_betas  = hmm_backward_msgs(log_A, log_likelihoods)          # (T, K)

    log_gamma = log_alphas + log_betas                              # (T, K)
    # Normalise each time step.
    log_gamma -= jax.nn.logsumexp(log_gamma, axis=1, keepdims=True)
    return jnp.exp(log_gamma)


@jit
def hmm_viterbi(
    log_pi0: jnp.ndarray,
    log_A: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
) -> jnp.ndarray:
    """Viterbi decoding: MAP state sequence via max-product.

    Args:
        log_pi0:         (K,)   Log initial-state probabilities.
        log_A:           (K, K) Log transition matrix.
        log_likelihoods: (T, K) Log observation likelihoods per state.

    Returns:
        states: (T,) MAP integer state sequence.
    """
    def fwd_step(carry, log_lik_t):
        log_delta = carry
        scores = log_delta[:, None] + log_A          # (K, K)
        log_delta_new = log_lik_t + jnp.max(scores, axis=0)   # (K,)
        psi_new      = jnp.argmax(scores, axis=0)              # (K,) backpointers
        return log_delta_new, (log_delta_new, psi_new)

    log_delta_0 = log_pi0 + log_likelihoods[0]
    _, (log_deltas_rest, psis) = lax.scan(fwd_step, log_delta_0, log_likelihoods[1:])

    # psis[t, k] = argmax over previous state for arriving in state k at t+1
    # Shape: (T-1, K)

    def bwd_step(z_next, psi_t):
        z_t = psi_t[z_next]
        return z_t, z_t

    z_T = jnp.argmax(
        jnp.concatenate([log_delta_0[None], log_deltas_rest], axis=0)[-1]
    )
    _, z_rest = lax.scan(bwd_step, z_T, psis[::-1])

    return jnp.concatenate([z_rest[::-1], jnp.array([z_T])])


# ---------------------------------------------------------------------------
# HSMM message passing
# ---------------------------------------------------------------------------

def hsmm_forward(
    log_pi0: jnp.ndarray,
    log_A: jnp.ndarray,
    log_dur: jnp.ndarray,
    log_likelihoods: jnp.ndarray,
    max_dur: int,
) -> jnp.ndarray:
    """Compute HSMM forward messages in log space.

    F[t, k] = log P(x_{1:t+1}, segment ends at t, state=k).

    Uses a circular buffer of the D most recent forward messages and
    pre-computed cumulative log-likelihoods for O(T·D·K²) runtime.

    Args:
        log_pi0:         (K,)         Log initial-state probabilities.
        log_A:           (K, K)       Log transition matrix.
                                      Self-transitions should be -inf for a
                                      proper HSMM (enforced externally).
        log_dur:         (K, max_dur) log_dur[k, d-1] = log P(dur=d | state=k).
        log_likelihoods: (T, K)       Log observation likelihoods per state.
        max_dur:         int          Maximum segment duration to consider.

    Returns:
        log_F: (T, K) Forward messages.
    """
    T, K = log_likelihoods.shape
    D = max_dur

    # cumlik[t, k] = Σ_{s=0}^{t-1} log p(x_s | k)   (cumlik[0, :] = 0)
    cumlik = jnp.concatenate(
        [jnp.zeros((1, K)), jnp.cumsum(log_likelihoods, axis=0)], axis=0
    )  # (T+1, K)

    log_dur_T = log_dur.T          # (D, K)  for broadcasting
    dur_idx   = jnp.arange(1, D + 1)  # (D,)

    # Initial buffer: "virtual" start segment ending just before t=0.
    # buf[d-1, :] represents log_F at time (current_t - d).
    log_F_buf_init = jnp.full((D, K), -jnp.inf)
    log_F_buf_init = log_F_buf_init.at[0].set(log_pi0)

    def step(log_F_buf, t):
        t1 = t + 1  # 1-indexed for cumlik

        # Segment log-likelihoods for all durations d=1..D ending at time t.
        # seg_ll[d-1, k] = cumlik[t1, k] - cumlik[t1-d, k]
        past_t = jnp.clip(t1 - dur_idx, 0, T)           # (D,)
        seg_ll  = cumlik[t1][None, :] - cumlik[past_t]   # (D, K)

        # Mask out durations that reach before t=0.
        valid   = (dur_idx <= t1)[:, None]               # (D, 1)
        seg_ll  = jnp.where(valid, seg_ll, -jnp.inf)

        # Transition contribution:
        # incoming[d-1, k] = logsumexp_j ( log_F[t-d, j] + log_A[j, k] )
        # log_F_buf[d-1, :] = log_F at (t - d)
        incoming = jax.nn.logsumexp(
            log_F_buf[:, :, None] + log_A[None, :, :],  # (D, K, K)
            axis=1,                                       # sum over j -> (D, K)
        )

        # Combine over durations.
        log_F_new = jax.nn.logsumexp(
            incoming + seg_ll + log_dur_T,               # (D, K)
            axis=0,                                       # sum over d -> (K,)
        )

        # Prepend new entry; drop the oldest.
        log_F_buf_new = jnp.concatenate(
            [log_F_new[None, :], log_F_buf[:-1, :]], axis=0
        )
        return log_F_buf_new, log_F_new

    step_jit = jit(step)
    _, log_Fs = lax.scan(step_jit, log_F_buf_init, jnp.arange(T))
    return log_Fs  # (T, K)


def hsmm_backward_sample(
    key,
    log_A: np.ndarray,
    log_dur: np.ndarray,
    log_F: np.ndarray,
    log_likelihoods: np.ndarray,
    max_dur: int,
) -> np.ndarray:
    """Sample a state sequence from HSMM forward messages.

    Uses a Python while-loop (not jit-compiled) because the number of
    segments is data-dependent.  Runtime is O(n_segments · D · K), which
    is fast since n_segments ≪ T in practice.

    Args:
        key:             JAX PRNGKey.
        log_A:           (K, K)         Log transition matrix (numpy).
        log_dur:         (K, max_dur)   Log duration probabilities (numpy).
        log_F:           (T, K)         Forward messages from hsmm_forward().
        log_likelihoods: (T, K)         Per-frame per-state log likelihoods.
        max_dur:         int            Maximum duration.

    Returns:
        states: (T,) numpy integer array of per-frame state assignments.
    """
    log_F   = np.array(log_F)
    log_A   = np.array(log_A)
    log_dur = np.array(log_dur)
    log_lik = np.array(log_likelihoods)

    T, K = log_F.shape
    states = np.zeros(T, dtype=np.int32)

    # Pre-compute cumulative log-likelihoods for segment scoring.
    cumlik = np.concatenate([np.zeros((1, K)), np.cumsum(log_lik, axis=0)], axis=0)

    t = T - 1  # current segment end (0-indexed, inclusive)

    # Sample the state of the final segment.
    key, subkey = jax.random.split(key)
    z = int(jax.random.categorical(subkey, jnp.array(log_F[t])))

    while t >= 0:
        max_d = min(max_dur, t + 1)
        d_range = np.arange(1, max_d + 1)

        # P(d | z, end_at_t, x) ∝ p(d|z) * seg_ll(t-d+1, t; z)
        #                         * logsumexp_j( log_F[t-d, j] + log_A[j, z] )
        seg_lls   = cumlik[t + 1, z] - cumlik[t + 1 - d_range, z]  # (max_d,)
        dur_lls   = log_dur[z, :max_d]                               # (max_d,)
        if t >= 1:
            past_F = log_F[t - d_range, :]                           # (max_d, K)
        else:
            past_F = np.full((max_d, K), -np.inf)
            past_F[max_d - 1, :] = 0.0   # virtual start at t=-1

        incoming  = np.logaddexp.reduce(
            past_F + log_A[:, z][None, :], axis=1            # (max_d,)
        )
        log_w = dur_lls + seg_lls + incoming

        key, subkey = jax.random.split(key)
        d = int(jax.random.categorical(subkey, jnp.array(log_w))) + 1

        # Fill in state assignments for this segment.
        seg_start = max(0, t - d + 1)
        states[seg_start : t + 1] = z

        # Move to the end of the preceding segment.
        t = seg_start - 1
        if t < 0:
            break

        # Sample the preceding state.
        key, subkey = jax.random.split(key)
        log_probs = jnp.array(log_F[t] + log_A[:, z])
        z = int(jax.random.categorical(subkey, log_probs))

    return states
