"""
Tests for jax_hsmm.ARHMM and jax_hsmm.ARHSMM.

Covers:
    - Basic fit / predict pipeline
    - Label shape and sentinel padding
    - Log-likelihood is finite and increases (on average) over training
    - Whitening: stored params, invertibility
    - Empirical Bayes prior initialization
    - Robust (Student-t) observations
    - Separate transition matrices per group
    - heldout_viterbi on new sessions
    - E-step (expected_states shape and normalization)
    - Checkpointing via get_labels / stateseqs / num_states properties
    - Transition sampler: count_transitions correctness
    - regularize_for_stability: does not crash on ill-conditioned data
    - ARHSMM: basic fit
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_hsmm import ARHMM, ARHSMM
from jax_hsmm.transitions import count_transitions, count_transitions_batch
from jax_hsmm.util import regularize_for_stability
from jax_hsmm.observations import (
    make_ar_features,
    ar_log_likelihoods,
    ar_log_likelihoods_student_t,
    compute_sufficient_stats,
    sample_obs_params,
    default_mniw_prior,
    empirical_bayes_mniw_prior,
)
from jax_hsmm.messages import (
    hmm_forward,
    hmm_backward_sample,
    hmm_viterbi,
    hmm_expected_states,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM  = 10
N_STATES = 15
AR_LAGS  = 3
N_ITER   = 10  # keep short for CI


def _make_data(n_sessions=4, T=600, D=OBS_DIM, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((T, D)).astype(np.float32)
            for _ in range(n_sessions)]


def _make_key(seed=0):
    return jax.random.PRNGKey(seed)


# ---------------------------------------------------------------------------
# 1. Basic fit
# ---------------------------------------------------------------------------

class TestBasicFit:

    def test_fit_returns_samples(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=0)
        samples, _ = model.fit(_make_key(), _make_data(), n_iter=N_ITER, verbose=False)
        assert len(samples) == N_ITER

    def test_samples_contain_expected_keys(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=0)
        samples, _ = model.fit(_make_key(), _make_data(), n_iter=N_ITER, verbose=False)
        for s in samples:
            assert 'A'     in s
            assert 'Sigma' in s
            assert 'pi'    in s
            assert 'beta'  in s

    def test_store_every(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=0)
        samples, _ = model.fit(_make_key(), _make_data(),
                               n_iter=N_ITER, store_every=2, verbose=False)
        assert len(samples) == N_ITER // 2

    def test_param_shapes(self):
        D, K, L = OBS_DIM, N_STATES, AR_LAGS
        D_phi = D * L + 1  # affine=True by default
        model = ARHMM(n_states=K, obs_dim=D, ar_lags=L, seed=0)
        samples, _ = model.fit(_make_key(), _make_data(), n_iter=2, verbose=False)
        s = samples[-1]
        assert s['A'].shape     == (K, D, D_phi)
        assert s['Sigma'].shape == (K, D, D)
        assert s['pi'].shape    == (K, K)
        assert s['beta'].shape  == (K,)

    def test_pi_rows_sum_to_one(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=0)
        samples, _ = model.fit(_make_key(), _make_data(), n_iter=5, verbose=False)
        pi = samples[-1]['pi']
        np.testing.assert_allclose(pi.sum(axis=1), np.ones(N_STATES), atol=1e-5)

    def test_beta_sums_to_one(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=0)
        samples, _ = model.fit(_make_key(), _make_data(), n_iter=5, verbose=False)
        beta = samples[-1]['beta']
        assert abs(beta.sum() - 1.0) < 1e-5

    def test_sigma_positive_definite(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=0)
        samples, _ = model.fit(_make_key(), _make_data(), n_iter=5, verbose=False)
        Sigma = samples[-1]['Sigma']
        for k in range(N_STATES):
            eigs = np.linalg.eigvalsh(Sigma[k])
            assert eigs.min() > 0, f"Sigma[{k}] not PD"


# ---------------------------------------------------------------------------
# 2. Labels and state sequences
# ---------------------------------------------------------------------------

class TestLabels:

    def setup_method(self):
        self.data = _make_data(n_sessions=3, T=500)
        self.model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM,
                           ar_lags=AR_LAGS, seed=1)
        self.model.fit(_make_key(1), self.data, n_iter=N_ITER, verbose=False)

    def test_stateseqs_length(self):
        seqs = self.model.stateseqs
        assert len(seqs) == 3
        for seq, d in zip(seqs, self.data):
            # state seq covers T - lag frames
            assert len(seq) == len(d) - AR_LAGS

    def test_get_labels_length(self):
        labels = self.model.get_labels()
        for lab, d in zip(labels, self.data):
            assert len(lab) == len(d)

    def test_get_labels_sentinel_prefix(self):
        labels = self.model.get_labels(sentinel=-5)
        for lab in labels:
            assert (lab[:AR_LAGS] == -5).all()

    def test_get_labels_valid_states(self):
        labels = self.model.get_labels()
        for lab in labels:
            body = lab[AR_LAGS:]
            assert body.min() >= 0
            assert body.max() < N_STATES

    def test_num_states_property(self):
        assert self.model.num_states == N_STATES


# ---------------------------------------------------------------------------
# 3. Log-likelihood
# ---------------------------------------------------------------------------

class TestLogLikelihood:

    def test_log_likelihood_finite(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=2)
        model.fit(_make_key(2), _make_data(), n_iter=N_ITER, verbose=False)
        ll = model.log_likelihood()
        assert np.isfinite(ll)

    def test_log_likelihood_negative(self):
        # Log-likelihood of a proper probability model should be <= 0
        # (summed over frames, not averaged)
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=2)
        model.fit(_make_key(2), _make_data(), n_iter=N_ITER, verbose=False)
        ll = model.log_likelihood()
        assert ll < 0

    def test_log_likelihood_increases(self):
        # Average LL should trend upward over training.
        # We check first vs last third rather than monotonicity
        # (Gibbs is stochastic).
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=3)
        data  = _make_data(n_sessions=3, T=800)
        n_iter = 30

        lls = []
        # Use store_every=1 and compute LL from samples directly
        samples, _ = model.fit(_make_key(3), data,
                               n_iter=n_iter, store_every=1, verbose=False)
        for s in samples:
            lls.append(model.log_likelihood(params=s, data_list=data))

        first_third = np.mean(lls[:n_iter//3])
        last_third  = np.mean(lls[-n_iter//3:])
        assert last_third > first_third, (
            f"LL did not improve: {first_third:.2f} → {last_third:.2f}"
        )

    def test_log_likelihood_with_explicit_data(self):
        data  = _make_data()
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS, seed=2)
        model.fit(_make_key(2), data, n_iter=5, verbose=False)
        ll_cached   = model.log_likelihood()
        ll_explicit = model.log_likelihood(data_list=data)
        # Should be close (both use same whitening)
        assert abs(ll_cached - ll_explicit) < 1.0


# ---------------------------------------------------------------------------
# 4. Whitening
# ---------------------------------------------------------------------------

class TestWhitening:

    def test_whitening_params_stored(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM,
                      ar_lags=AR_LAGS, whiten='all', seed=0)
        model.fit(_make_key(), _make_data(), n_iter=2, verbose=False)
        wp = model.whitening_parameters
        assert wp is not None
        assert wp['mu'].shape == (OBS_DIM,)
        assert wp['L'].shape  == (OBS_DIM, OBS_DIM)

    def test_no_whitening_params_when_disabled(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM,
                      ar_lags=AR_LAGS, whiten='none', seed=0)
        model.fit(_make_key(), _make_data(), n_iter=2, verbose=False)
        assert model.whitening_parameters is None

    def test_unwhiten_roundtrip(self):
        data  = _make_data(n_sessions=1, T=300)
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM,
                      ar_lags=AR_LAGS, whiten='all', seed=0)
        model.fit(_make_key(), data, n_iter=2, verbose=False)

        # Whiten then unwhiten should recover original data
        whitened   = model._whiten_new_data(data)
        recovered  = [model.unwhiten(w) for w in whitened]
        np.testing.assert_allclose(
            recovered[0], data[0].astype(np.float64), atol=1e-5
        )


# ---------------------------------------------------------------------------
# 5. Empirical Bayes
# ---------------------------------------------------------------------------

class TestEmpiricalBayes:

    def test_eb_prior_psi0_shape(self):
        data  = _make_data()
        prior = empirical_bayes_mniw_prior(data, OBS_DIM, AR_LAGS, affine=True)
        assert prior['Psi_0'].shape == (OBS_DIM, OBS_DIM)
        assert prior['M_0'].shape   == (OBS_DIM, OBS_DIM * AR_LAGS + 1)

    def test_eb_prior_psi0_positive_definite(self):
        data  = _make_data()
        prior = empirical_bayes_mniw_prior(data, OBS_DIM, AR_LAGS, affine=True)
        eigs  = np.linalg.eigvalsh(prior['Psi_0'])
        assert eigs.min() > 0

    def test_eb_fit_does_not_crash(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                      empirical_bayes=True, seed=0)
        model.fit(_make_key(), _make_data(), n_iter=3, verbose=False)


# ---------------------------------------------------------------------------
# 6. Robust (Student-t) observations
# ---------------------------------------------------------------------------

class TestRobust:

    def test_robust_fit_does_not_crash(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                      robust=True, nu=3.0, seed=4)
        model.fit(_make_key(4), _make_data(), n_iter=N_ITER, verbose=False)

    def test_robust_labels_valid(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                      robust=True, seed=4)
        model.fit(_make_key(4), _make_data(), n_iter=5, verbose=False)
        labels = model.get_labels()
        for lab in labels:
            assert (lab[AR_LAGS:] >= 0).all()
            assert (lab[AR_LAGS:] < N_STATES).all()

    def test_robust_ll_finite(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                      robust=True, seed=4)
        model.fit(_make_key(4), _make_data(), n_iter=5, verbose=False)
        ll = model.log_likelihood()
        assert np.isfinite(ll)

    def test_student_t_ll_less_sensitive_to_outliers(self):
        # Inject outlier frames — Student-t LL should be higher than Gaussian
        rng  = np.random.default_rng(5)
        data = _make_data(n_sessions=2, T=400, seed=5)
        # Add large outliers to 5% of frames
        for d in data:
            idx = rng.choice(len(d), size=len(d)//20, replace=False)
            d[idx] *= 20.0

        D_phi = OBS_DIM * AR_LAGS + 1
        K     = N_STATES
        prior = default_mniw_prior(OBS_DIM, AR_LAGS, affine=True)

        # Fit a single set of params from prior for fair comparison
        rng_np = np.random.default_rng(5)
        dummy_stats = compute_sufficient_stats(
            jnp.zeros((1, D_phi)), jnp.zeros((1, OBS_DIM)),
            jnp.zeros(1, dtype=jnp.int32), K
        )
        A, Sigma = sample_obs_params(dummy_stats, prior, rng_np)
        phi, x   = make_ar_features(data[0], AR_LAGS, affine=True)
        phi_j, x_j = jnp.array(phi), jnp.array(x)

        ll_gaussian = float(ar_log_likelihoods(phi_j, x_j, A, Sigma).mean())
        ll_student  = float(ar_log_likelihoods_student_t(
            phi_j, x_j, A, Sigma, nu=3.0).mean())

        # Student-t should assign higher LL to outlier-contaminated data
        # (less extreme tails = less penalty for outliers)
        assert ll_student > ll_gaussian


# ---------------------------------------------------------------------------
# 7. Separate transition matrices
# ---------------------------------------------------------------------------

class TestSeparateTrans:

    def _fit_separate(self, n_sessions=6):
        data      = _make_data(n_sessions=n_sessions, T=500)
        group_ids = ['ctrl'] * (n_sessions // 2) + ['ko'] * (n_sessions // 2)
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                      separate_trans=True, seed=6)
        samples, _ = model.fit(_make_key(6), data, n_iter=N_ITER,
                               verbose=False, group_ids=group_ids)
        return model, samples, group_ids

    def test_separate_trans_fit_does_not_crash(self):
        self._fit_separate()

    def test_separate_trans_requires_group_ids(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                      separate_trans=True, seed=6)
        with pytest.raises(ValueError, match="group_ids"):
            model.fit(_make_key(6), _make_data(), n_iter=2, verbose=False)

    def test_separate_trans_pi_groups_in_samples(self):
        _, samples, _ = self._fit_separate()
        for s in samples:
            assert 'pi_groups' in s
            assert 'ctrl' in s['pi_groups']
            assert 'ko'   in s['pi_groups']

    def test_separate_trans_pi_rows_sum_to_one(self):
        _, samples, _ = self._fit_separate()
        for g in ('ctrl', 'ko'):
            pi = samples[-1]['pi_groups'][g]
            np.testing.assert_allclose(
                pi.sum(axis=1), np.ones(N_STATES), atol=1e-5
            )

    def test_separate_trans_groups_differ(self):
        # Verify per-group counts are accumulated separately (matrices differ).
        _, samples, _ = self._fit_separate(n_sessions=8)
        pi_ctrl = samples[-1]['pi_groups']['ctrl']
        pi_ko   = samples[-1]['pi_groups']['ko']
        assert not np.array_equal(pi_ctrl, pi_ko)


# ---------------------------------------------------------------------------
# 8. heldout_viterbi
# ---------------------------------------------------------------------------

class TestHeldoutViterbi:

    def setup_method(self):
        self.data  = _make_data(n_sessions=3, T=500)
        self.model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM,
                           ar_lags=AR_LAGS, seed=7)
        self.model.fit(_make_key(7), self.data, n_iter=N_ITER, verbose=False)

    def test_heldout_viterbi_length(self):
        new = _make_data(n_sessions=1, T=400, seed=99)[0]
        states = self.model.heldout_viterbi(new)
        assert len(states) == 400 - AR_LAGS

    def test_heldout_viterbi_valid_states(self):
        new    = _make_data(n_sessions=1, T=400, seed=99)[0]
        states = self.model.heldout_viterbi(new)
        assert states.min() >= 0
        assert states.max() < N_STATES

    def test_heldout_viterbi_requires_fit(self):
        model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM, ar_lags=AR_LAGS)
        new   = _make_data(n_sessions=1, T=200)[0]
        with pytest.raises(RuntimeError, match="fit"):
            model.heldout_viterbi(new)

    def test_viterbi_batch(self):
        new_data = _make_data(n_sessions=2, T=300, seed=88)
        results  = self.model.viterbi(self.model._params, new_data)
        assert len(results) == 2
        for r, d in zip(results, new_data):
            assert len(r) == len(d) - AR_LAGS


# ---------------------------------------------------------------------------
# 9. E-step
# ---------------------------------------------------------------------------

class TestEStep:

    def setup_method(self):
        self.data  = _make_data(n_sessions=2, T=400)
        self.model = ARHMM(n_states=N_STATES, obs_dim=OBS_DIM,
                           ar_lags=AR_LAGS, seed=8)
        self.model.fit(_make_key(8), self.data, n_iter=5, verbose=False)

    def test_expected_states_shape(self):
        gamma = self.model._E_step()
        assert len(gamma) == 2
        for g, d in zip(gamma, self.data):
            assert g.shape == (len(d) - AR_LAGS, N_STATES)

    def test_expected_states_sum_to_one(self):
        gamma = self.model._E_step()
        for g in gamma:
            np.testing.assert_allclose(g.sum(axis=1), np.ones(len(g)), atol=1e-5)

    def test_expected_states_non_negative(self):
        gamma = self.model._E_step()
        for g in gamma:
            assert (g >= 0).all()

    def test_expected_states_stored(self):
        self.model._E_step()
        assert self.model.expected_states is not None


# ---------------------------------------------------------------------------
# 10. Message passing primitives
# ---------------------------------------------------------------------------

class TestMessagePassing:

    def setup_method(self):
        K, T = 5, 50
        rng  = np.random.default_rng(0)
        # Random log-transition matrix (rows log-normalized)
        pi      = rng.dirichlet(np.ones(K), size=K)
        self.log_A    = jnp.log(jnp.array(pi) + 1e-300)
        self.log_pi0  = jnp.log(jnp.ones(K) / K)
        self.log_liks = jnp.array(
            rng.standard_normal((T, K)).astype(np.float32)
        )
        self.K = K
        self.T = T

    def test_forward_shape(self):
        alphas = hmm_forward(self.log_pi0, self.log_A, self.log_liks)
        assert alphas.shape == (self.T, self.K)

    def test_forward_finite(self):
        alphas = hmm_forward(self.log_pi0, self.log_A, self.log_liks)
        assert jnp.isfinite(alphas).all()

    def test_backward_sample_shape(self):
        alphas = hmm_forward(self.log_pi0, self.log_A, self.log_liks)
        states = hmm_backward_sample(_make_key(), self.log_A, alphas)
        assert states.shape == (self.T,)

    def test_backward_sample_valid_states(self):
        alphas = hmm_forward(self.log_pi0, self.log_A, self.log_liks)
        states = hmm_backward_sample(_make_key(), self.log_A, alphas)
        assert int(states.min()) >= 0
        assert int(states.max()) < self.K

    def test_viterbi_shape(self):
        states = hmm_viterbi(self.log_pi0, self.log_A, self.log_liks)
        assert states.shape == (self.T,)

    def test_expected_states_shape(self):
        gamma = hmm_expected_states(self.log_pi0, self.log_A, self.log_liks)
        assert gamma.shape == (self.T, self.K)

    def test_expected_states_normalized(self):
        gamma = hmm_expected_states(self.log_pi0, self.log_A, self.log_liks)
        np.testing.assert_allclose(
            np.array(gamma.sum(axis=1)), np.ones(self.T), atol=1e-5
        )

    def test_log_likelihood_from_forward(self):
        alphas = hmm_forward(self.log_pi0, self.log_A, self.log_liks)
        ll     = float(jax.nn.logsumexp(alphas[-1]))
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# 11. Transition counting
# ---------------------------------------------------------------------------

class TestTransitionCounting:

    def test_count_transitions_simple(self):
        states = np.array([0, 1, 0, 2, 0], dtype=np.int32)
        counts = count_transitions(states, K=3)
        assert counts[0, 1] == 1
        assert counts[1, 0] == 1
        assert counts[0, 2] == 1
        assert counts[2, 0] == 1
        assert counts.sum() == 4

    def test_count_transitions_self(self):
        states = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        counts = count_transitions(states, K=2)
        assert counts[0, 0] == 2
        assert counts[0, 1] == 1
        assert counts[1, 1] == 1
        assert counts.sum()  == 4

    def test_count_transitions_batch(self):
        seqs   = [np.array([0, 1, 0]), np.array([1, 2, 1])]
        counts = count_transitions_batch(seqs, K=3)
        assert counts[0, 1] == 1
        assert counts[1, 0] == 1
        assert counts[1, 2] == 1
        assert counts[2, 1] == 1

    def test_count_transitions_single_frame(self):
        states = np.array([3], dtype=np.int32)
        counts = count_transitions(states, K=5)
        assert counts.sum() == 0

    def test_count_transitions_shape(self):
        states = np.random.randint(0, 10, size=1000, dtype=np.int32)
        counts = count_transitions(states, K=10)
        assert counts.shape == (10, 10)
        assert counts.sum()  == 999   # T-1 transitions


# ---------------------------------------------------------------------------
# 12. regularize_for_stability
# ---------------------------------------------------------------------------

class TestRegularize:

    def _make_ill_conditioned_stats(self, K=5, D=4, D_phi=13):
        """Create a stats dict where some states have singular S_yy."""
        rng = np.random.default_rng(42)
        # State 0: well-conditioned
        # State 1: rank-deficient S_yy (will need regularization)
        n    = np.array([100.0, 50.0] + [0.0] * (K - 2))
        S_xx = np.zeros((K, D, D))
        S_xy = np.zeros((K, D, D_phi))
        S_yy = np.zeros((K, D_phi, D_phi))

        for k in range(2):
            phi = rng.standard_normal((int(n[k]), D_phi))
            x   = rng.standard_normal((int(n[k]), D))
            S_xx[k] = x.T @ x
            S_xy[k] = x.T @ phi
            S_yy[k] = phi.T @ phi

        # Make state 1 ill-conditioned by zeroing most of S_yy
        S_yy[1] *= 1e-10

        return dict(
            n    = jnp.array(n),
            S_xx = jnp.array(S_xx),
            S_xy = jnp.array(S_xy),
            S_yy = jnp.array(S_yy),
        )

    def test_does_not_crash(self):
        D, D_phi = 4, 13
        stats  = self._make_ill_conditioned_stats(D=D, D_phi=D_phi)
        prior  = default_mniw_prior(D, 3, affine=True)
        result = regularize_for_stability(stats, prior, verbose=False)
        assert result is not None

    def test_returns_valid_S_yy(self):
        D, D_phi = 4, 13
        stats  = self._make_ill_conditioned_stats(D=D, D_phi=D_phi)
        prior  = default_mniw_prior(D, 3, affine=True)
        result = regularize_for_stability(stats, prior, verbose=False)
        S_yy   = np.array(result['S_yy'])
        # All non-empty states should have PD S_yy after regularization
        for k in range(2):
            eigs = np.linalg.eigvalsh(S_yy[k])
            assert eigs.min() > 0, f"S_yy[{k}] not PD after regularization"

    def test_unchanged_when_already_pd(self):
        """If all states are already well-conditioned, return same dict."""
        D, D_phi, K = 4, 13, 3
        rng  = np.random.default_rng(0)
        n    = np.array([200.0] * K)
        S_yy = np.zeros((K, D_phi, D_phi))
        for k in range(K):
            phi     = rng.standard_normal((200, D_phi))
            S_yy[k] = phi.T @ phi + 10 * np.eye(D_phi)   # guaranteed PD

        stats = dict(
            n    = jnp.array(n),
            S_xx = jnp.zeros((K, D, D)),
            S_xy = jnp.zeros((K, D, D_phi)),
            S_yy = jnp.array(S_yy),
        )
        prior  = default_mniw_prior(D, 3, affine=True)
        result = regularize_for_stability(stats, prior, verbose=False)
        # Same object returned when nothing needed fixing
        assert result is stats


# ---------------------------------------------------------------------------
# 13. AR feature construction
# ---------------------------------------------------------------------------

class TestARFeatures:

    def test_shape_no_affine(self):
        T, D, lag = 100, 10, 3
        data = np.random.randn(T, D).astype(np.float32)
        phi, x = make_ar_features(data, lag, affine=False)
        assert phi.shape == (T - lag, D * lag)
        assert x.shape   == (T - lag, D)

    def test_shape_affine(self):
        T, D, lag = 100, 10, 3
        data = np.random.randn(T, D).astype(np.float32)
        phi, x = make_ar_features(data, lag, affine=True)
        assert phi.shape == (T - lag, D * lag + 1)
        assert x.shape   == (T - lag, D)

    def test_affine_column_is_ones(self):
        data = np.random.randn(50, 5).astype(np.float32)
        phi, _ = make_ar_features(data, lag=2, affine=True)
        np.testing.assert_array_equal(phi[:, -1], np.ones(48))

    def test_lag1_identity_recovery(self):
        # With lag=1 and no noise, phi should just be data[:-1]
        data = np.random.randn(20, 3).astype(np.float32)
        phi, x = make_ar_features(data, lag=1, affine=False)
        np.testing.assert_array_equal(phi, data[:-1])
        np.testing.assert_array_equal(x,   data[1:])


# ---------------------------------------------------------------------------
# 14. ARHSMM basic smoke test
# ---------------------------------------------------------------------------

class TestARHSMM:

    def test_arhsmm_fit_does_not_crash(self):
        model = ARHSMM(n_states=8, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                       max_dur=50, expected_dur=15.0, seed=9)
        model.fit(_make_key(9), _make_data(n_sessions=2, T=300),
                  n_iter=5, verbose=False)

    def test_arhsmm_labels_valid(self):
        model = ARHSMM(n_states=8, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                       max_dur=50, seed=9)
        model.fit(_make_key(9), _make_data(n_sessions=2, T=300),
                  n_iter=5, verbose=False)
        labels = model.get_labels()
        for lab in labels:
            assert (lab[AR_LAGS:] >= 0).all()
            assert (lab[AR_LAGS:] < 8).all()

    def test_arhsmm_lam_shape(self):
        model = ARHSMM(n_states=8, obs_dim=OBS_DIM, ar_lags=AR_LAGS,
                       max_dur=50, seed=9)
        samples, _ = model.fit(_make_key(9), _make_data(n_sessions=2, T=300),
                               n_iter=3, verbose=False)
        assert samples[-1]['lam'].shape == (8,)
        assert (samples[-1]['lam'] > 0).all()
