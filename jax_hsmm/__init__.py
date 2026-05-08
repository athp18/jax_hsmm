"""
jax_hsmm — JAX-native AR-HMM / AR-HSMM replacing pyhsmm.

Modules
-------
messages      JAX forward-backward message passing (jit + lax.scan)
observations  AR-Gaussian observation model with MNIW conjugate updates
transitions   Sticky HDP-HMM transition Gibbs sampler
durations     Poisson duration distribution for HSMM
model         Top-level ARHMM and ARHSMM classes

Quick start
-----------
    import jax
    from jax_hsmm import ARHMM, ARHSMM

    key   = jax.random.PRNGKey(0)
    model = ARHSMM(n_states=100, obs_dim=10, ar_lags=3, max_dur=100)
    samples = model.fit(key, data_list, n_iter=200)

    labels = model.viterbi(samples[-1], data_list)
"""

from jax_hsmm.model import ARHMM, ARHSMM, whiten_data
from jax_hsmm.observations import (
    empirical_bayes_mniw_prior,
    default_mniw_prior,
    ar_log_likelihoods_student_t,
    compute_weighted_sufficient_stats,
    sample_robust_weights,
)
from jax_hsmm.messages import hmm_expected_states, hmm_backward_msgs
from jax_hsmm.transitions import sample_transitions_separate
from jax_hsmm.util import regularize_for_stability

__all__ = [
    'ARHMM', 'ARHSMM',
    'whiten_data',
    'empirical_bayes_mniw_prior', 'default_mniw_prior',
    'ar_log_likelihoods_student_t',
    'compute_weighted_sufficient_stats',
    'sample_robust_weights',
    'hmm_expected_states', 'hmm_backward_msgs',
    'sample_transitions_separate',
    'regularize_for_stability',
]
