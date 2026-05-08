"""
Top-level AR-HMM and AR-HSMM model classes.

Both classes share the same Gibbs sweep structure:

    (1) State-sequence sampling  — forward-backward via JAX message passing
    (2) Observation params       — MNIW conjugate update (scipy)
    (3) Transition params        — sticky HDP-HMM update (numpy)
    (4) [HSMM only] Duration params — Poisson-Gamma conjugate update

Key options (mirrors moseq2-model behaviour)
--------------------------------------------
    affine=True           Bias column appended to phi (moseq2-model default).
    whiten=True           Cholesky-whiten all sessions using pooled covariance.
    empirical_bayes=True  Set Psi_0 from a global AR MLE fit (moseq2-model EB init).
    separate_trans=True   Each experimental group gets its own transition matrix
                          pi_g; all groups share beta and observation parameters.
                          Requires group_ids to be passed to fit().
    robust=True           Student-t observation noise: auxiliary precision weights
                          tau_t ~ Gamma(nu/2, nu/2) are Gibbs-sampled each sweep
                          and weight the MNIW sufficient statistics.

Usage
-----
    import jax, numpy as np
    from jax_hsmm.model import ARHMM

    model = ARHMM(n_states=100, obs_dim=10, ar_lags=3,
                  affine=True, whiten=True, empirical_bayes=True)
    samples = model.fit(key, data_list, n_iter=200)

    # with separate transition matrices
    model_st = ARHMM(n_states=100, obs_dim=10, ar_lags=3, separate_trans=True)
    samples = model_st.fit(key, data_list, n_iter=200,
                           group_ids=['ctrl', 'ctrl', 'ko', 'ko'])
"""

import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from jax_hsmm.messages import (
    hmm_forward,
    hmm_backward_sample,
    hmm_backward_msgs,
    hmm_expected_states,
    hmm_viterbi,
    hsmm_forward,
    hsmm_backward_sample,
)
from jax_hsmm.observations import (
    make_ar_features,
    ar_log_likelihoods,
    ar_log_likelihoods_student_t,
    compute_sufficient_stats,
    compute_weighted_sufficient_stats,
    sample_obs_params,
    sample_robust_weights,
    default_mniw_prior,
    empirical_bayes_mniw_prior,
)
from jax_hsmm.transitions import (
    sample_transitions,
    sample_transitions_separate,
    init_transitions,
)
from jax_hsmm.durations import (
    poisson_log_dur_matrix,
    sample_duration_params,
    default_duration_prior,
)
from jax_hsmm.util import regularize_for_stability


# ---------------------------------------------------------------------------
# Data whitening
# ---------------------------------------------------------------------------

def whiten_data(data_list):
    """Cholesky-whiten all sessions using the pooled global covariance.

    Mirrors moseq2-model's ``whiten_all``.

    Returns:
        whitened:          List of (T_i, D) float32 whitened arrays.
        whitening_params:  Dict with keys 'mu', 'L', 'offset'.
    """
    all_data = np.concatenate(
        [np.asarray(d, dtype=np.float64) for d in data_list], axis=0
    )
    mu  = all_data.mean(axis=0)
    cov = np.cov(all_data, rowvar=False, bias=True)

    min_eig = np.linalg.eigvalsh(cov).min()
    if min_eig < 1e-8:
        cov += (1e-8 - min_eig) * np.eye(cov.shape[0])

    L      = np.linalg.cholesky(cov)
    offset = 0.0

    def _apply(d):
        return np.linalg.solve(L, (np.asarray(d, np.float64) - mu).T).T + offset

    whitened = [_apply(d).astype(np.float32) for d in data_list]
    return whitened, {'mu': mu, 'L': L, 'offset': offset}


def whiten_data_each(data_list):
    """Cholesky-whiten each session independently.

    Mirrors moseq2-model's ``whiten_each``.

    Returns:
        whitened:          List of (T_i, D) float32 whitened arrays.
        whitening_params:  Dict mapping session index -> {'mu', 'L', 'offset'}.
    """
    whitened         = []
    whitening_params = {}
    for i, d in enumerate(data_list):
        w_list, wp = whiten_data([d])
        whitened.append(w_list[0])
        whitening_params[i] = wp
    return whitened, whitening_params


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _init_states_random(rng, data_list, K, lag):
    return [rng.integers(0, K, size=d.shape[0] - lag) for d in data_list]


def _prep_data(data_list, lag, affine):
    return [
        make_ar_features(np.array(d, dtype=np.float32), lag, affine=affine)
        for d in data_list
    ]


def _store_params(params):
    """Deep-copy params dict to numpy, handling nested pi_groups dict."""
    stored = {}
    for k, v in params.items():
        if k == 'pi_groups':
            stored[k] = {g: np.array(pi_g) for g, pi_g in v.items()}
        else:
            stored[k] = np.array(v)
    return stored


def _get_log_A(params, group_id, separate_trans):
    """Return log-transition matrix for a session given its group."""
    if separate_trans and 'pi_groups' in params and group_id is not None:
        pi = params['pi_groups'][group_id]
    else:
        pi = params['pi']
    return jnp.log(jnp.array(pi) + 1e-300)


# ---------------------------------------------------------------------------
# AR-HMM
# ---------------------------------------------------------------------------

class ARHMM:
    """Autoregressive HMM with sticky HDP-HMM transition prior.

    Direct JAX replacement for moseq2-model's FastARWeakLimitStickyHDPHMM.
    Duration distribution is implicit (geometric) via kappa-inflated
    self-transitions — no explicit duration model.

    Parameters
    ----------
    n_states : int
    obs_dim : int
    ar_lags : int
    alpha : float          HDP row concentration.
    kappa : float          Stickiness (self-transition) bonus. Default 1e6
                           matches moseq2-model.
    gamma : float          HDP top-level concentration.
    affine : bool          Append bias column to phi.
    whiten : bool          Cholesky-whiten data before fitting.
    empirical_bayes : bool Set Psi_0 from global AR MLE fit.
    separate_trans : bool  Per-group transition matrices.  Requires group_ids
                           in fit().
    robust : bool          Student-t observation noise (nu degrees of freedom).
    nu : float             Degrees of freedom for robust model (default 3).
    seed : int
    """

    def __init__(
        self,
        n_states: int        = 100,
        obs_dim: int         = 10,
        ar_lags: int         = 3,
        alpha: float         = 5.7,
        kappa: float         = 1e6,
        gamma: float         = 999.0,
        affine: bool         = True,
        whiten: str          = 'all',
        empirical_bayes: bool = True,
        separate_trans: bool = False,
        robust: bool         = False,
        nu: float            = 3.0,
        seed: int            = 0,
    ):
        self.K               = n_states
        self.D               = obs_dim
        self.lag             = ar_lags
        self.nlags           = ar_lags
        self.affine          = affine
        self.whiten_mode     = (whiten or 'none').lower()
        self.empirical_bayes = empirical_bayes
        self.separate_trans  = separate_trans
        self.robust          = robust
        self.nu              = nu
        self.rng             = np.random.default_rng(seed)

        self.whitening_parameters = None  # set during fit(); dict for 'all', dict-of-dicts for 'each'

        self.obs_prior = default_mniw_prior(obs_dim, ar_lags, affine=affine)

        self.trans_params, _pi0, _beta0 = init_transitions(
            n_states, alpha, kappa, gamma
        )

        # State / stat storage updated after every Gibbs sweep.
        self.state_seqs      = None
        self._obs_stats      = None
        self._params         = None
        self._prepped_data   = None
        self._group_ids      = None
        self.expected_states = None

    # ------------------------------------------------------------------
    # Whitening helpers
    # ------------------------------------------------------------------

    def _apply_whitening(self, data_list):
        if self.whiten_mode == 'all':
            whitened, wp = whiten_data(data_list)
            self.whitening_parameters = wp
        elif self.whiten_mode == 'each':
            whitened, wp = whiten_data_each(data_list)
            self.whitening_parameters = wp
        else:
            whitened = data_list
            self.whitening_parameters = None
        return whitened

    def _whiten_new_data(self, data_list):
        """Apply stored whitening to new data.

        For 'each' mode applied to genuinely new sessions (not in the stored
        per-session index), falls back to session 0's params — same behaviour
        as moseq2-model's apply_model which always uses a single stored dict.
        """
        if self.whitening_parameters is None:
            return data_list
        if self.whiten_mode == 'all':
            wp = self.whitening_parameters
            mu, L, offset = wp['mu'], wp['L'], wp['offset']
            return [
                (np.linalg.solve(L, (np.asarray(d, np.float64) - mu).T).T
                 + offset).astype(np.float32)
                for d in data_list
            ]
        elif self.whiten_mode == 'each':
            result = []
            fallback = self.whitening_parameters[0]
            for i, d in enumerate(data_list):
                wp = self.whitening_parameters.get(i, fallback)
                mu, L, offset = wp['mu'], wp['L'], wp['offset']
                result.append(
                    (np.linalg.solve(L, (np.asarray(d, np.float64) - mu).T).T
                     + offset).astype(np.float32)
                )
            return result
        return data_list

    def unwhiten(self, data, session_idx=None):
        """Invert the whitening transform on a (T, D) array.

        Args:
            data:        (T, D) whitened array.
            session_idx: For whiten='each', the original session index.
        """
        if self.whitening_parameters is None:
            return data
        if self.whiten_mode == 'all':
            wp = self.whitening_parameters
        elif self.whiten_mode == 'each':
            wp = self.whitening_parameters.get(session_idx or 0)
        else:
            return data
        mu, L = wp['mu'], wp['L']
        return (np.asarray(data, np.float64) @ L.T) + mu

    # ------------------------------------------------------------------
    # Core Gibbs step
    # ------------------------------------------------------------------

    def _gibbs_step(self, key, params, prepped_data, group_ids=None):
        """One full Gibbs sweep.

        Args:
            key:          JAX PRNGKey.
            params:       Current parameter dict.
            prepped_data: List of (phi, x) tuples.
            group_ids:    List of group labels (one per session); required
                          when separate_trans=True.

        Returns:
            new_params, state_seqs
        """
        A, Sigma = params['A'], params['Sigma']
        beta     = params['beta']

        log_pi0 = jnp.log(jnp.array(beta) + 1e-300)

        # ---- (1) Sample state sequences (forward-backward) ----
        state_seqs = []
        all_phi    = []
        all_x      = []

        for i, (phi, x) in enumerate(prepped_data):
            g_id  = group_ids[i] if group_ids is not None else None
            log_A = _get_log_A(params, g_id, self.separate_trans)

            phi_j = jnp.array(phi)
            x_j   = jnp.array(x)

            if self.robust:
                log_liks = ar_log_likelihoods_student_t(
                    phi_j, x_j, A, Sigma, self.nu
                )
            else:
                log_liks = ar_log_likelihoods(phi_j, x_j, A, Sigma)

            log_alphas = hmm_forward(log_pi0, log_A, log_liks)
            key, subkey = jax.random.split(key)
            states = hmm_backward_sample(subkey, log_A, log_alphas)
            state_seqs.append(np.array(states))
            all_phi.append(phi_j)
            all_x.append(x_j)

        # ---- (2) Sample observation parameters ----
        phi_cat    = jnp.concatenate(all_phi,    axis=0)
        x_cat      = jnp.concatenate(all_x,      axis=0)
        states_cat = jnp.array(np.concatenate(state_seqs, axis=0))

        if self.robust:
            tau = sample_robust_weights(
                self.rng,
                np.array(phi_cat), np.array(x_cat),
                np.array(states_cat),
                np.array(A), np.array(Sigma),
                self.nu,
            )
            stats = compute_weighted_sufficient_stats(
                phi_cat, x_cat, states_cat, self.K,
                jnp.array(tau, dtype=jnp.float32),
            )
        else:
            stats = compute_sufficient_stats(phi_cat, x_cat, states_cat, self.K)

        # ---- (2a) Regularise stats for numerical stability ----
        # Gate on self._obs_stats is not None: skip the very first sweep where
        # stats come from dummy data (mirrors moseq2-model's train_model gate).
        if self._obs_stats is not None:
            stats = regularize_for_stability(stats, self.obs_prior)

        A_new, Sigma_new = sample_obs_params(stats, self.obs_prior, self.rng)

        # ---- (3) Sample transition parameters ----
        if self.separate_trans and group_ids is not None:
            seqs_by_group: dict = {}
            for seqs, g_id in zip(state_seqs, group_ids):
                seqs_by_group.setdefault(g_id, []).append(seqs)

            pi_groups_new, beta_new = sample_transitions_separate(
                self.rng, seqs_by_group, beta, self.trans_params
            )
            # Store a pooled pi (mean over groups) for API compatibility and
            # use in sessions whose group is not in pi_groups.
            pi_pooled = np.mean(list(pi_groups_new.values()), axis=0)
            new_params = dict(
                A=A_new, Sigma=Sigma_new,
                pi=pi_pooled, pi_groups=pi_groups_new,
                beta=beta_new,
            )
        else:
            pi_new, beta_new = sample_transitions(
                self.rng, state_seqs, beta, self.trans_params
            )
            new_params = dict(A=A_new, Sigma=Sigma_new, pi=pi_new, beta=beta_new)

        # Cache for external access (mirrors moseq2-model's API).
        self.state_seqs = state_seqs
        self._obs_stats = stats
        self._params    = new_params

        return new_params, state_seqs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_params(self, key, data_list, group_ids=None):
        """Initialise model parameters from the prior.

        Args:
            key:       JAX PRNGKey.
            data_list: List of (T_i, D) arrays (already whitened if whiten=True).
            group_ids: List of group labels; required when separate_trans=True.

        Returns:
            params dict.
        """
        D_phi = self.D * self.lag + (1 if self.affine else 0)

        dummy_phi    = jnp.zeros((1, D_phi))
        dummy_x      = jnp.zeros((1, self.D))
        dummy_states = jnp.zeros((1,), dtype=jnp.int32)
        stats = compute_sufficient_stats(dummy_phi, dummy_x, dummy_states, self.K)
        A, Sigma = sample_obs_params(stats, self.obs_prior, self.rng)

        _, pi, beta = init_transitions(
            self.K,
            self.trans_params['alpha'],
            self.trans_params['kappa'],
            self.trans_params['gamma'],
        )

        params = dict(A=A, Sigma=Sigma, pi=pi, beta=beta)

        if self.separate_trans and group_ids is not None:
            unique_groups = sorted(set(group_ids), key=str)
            params['pi_groups'] = {
                g: self.rng.dirichlet(np.ones(self.K), size=self.K)
                for g in unique_groups
            }

        return params

    def fit(
        self,
        key,
        data_list,
        n_iter=200,
        verbose=True,
        store_every=1,
        group_ids=None,
        checkpoint_freq=None,
        checkpoint_path=None,
    ):
        """Run the Gibbs sampler.

        Args:
            key:             JAX PRNGKey.
            data_list:       List of (T_i, obs_dim) float32 arrays.
            n_iter:          Number of Gibbs iterations.
            verbose:         Show tqdm progress bar.
            store_every:     Store a snapshot every this many iterations.
            group_ids:       List of group labels, one per session.  Required when
                             separate_trans=True.  Labels can be any hashable.
            checkpoint_freq: Save a checkpoint every N iterations (None to disable).
            checkpoint_path: Directory to write checkpoint files into.

        Returns:
            samples: List of parameter dicts (one per stored iteration).
        """
        if self.separate_trans and group_ids is None:
            raise ValueError(
                "separate_trans=True requires group_ids (list of group labels, "
                "one per session in data_list)."
            )

        self._group_ids = group_ids

        if self.whiten_mode in ('all', 'each'):
            data_list = self._apply_whitening(data_list)

        if self.empirical_bayes:
            self.obs_prior = empirical_bayes_mniw_prior(
                data_list, self.D, self.lag, affine=self.affine
            )

        prepped = _prep_data(data_list, self.lag, self.affine)
        self._prepped_data = prepped

        params  = self.init_params(key, data_list, group_ids=group_ids)
        samples = []

        if checkpoint_freq is not None and checkpoint_path is not None:
            os.makedirs(checkpoint_path, exist_ok=True)

        for i in tqdm(range(n_iter), disable=not verbose, desc='Gibbs'):
            key, subkey = jax.random.split(key)
            params, _ = self._gibbs_step(
                subkey, params, prepped, group_ids=group_ids
            )
            if (i + 1) % store_every == 0:
                samples.append(_store_params(params))

            if (checkpoint_freq is not None
                    and checkpoint_path is not None
                    and (i + 1) % checkpoint_freq == 0):
                self._save_checkpoint(i + 1, params, samples, checkpoint_path)

        return samples

    def _save_checkpoint(self, itr, params, samples, checkpoint_path):
        """Save a checkpoint to disk.

        Format mirrors moseq2-model's save_arhmm_checkpoint:
            iter, model, params, samples, log_likelihood, labels.
        """
        checkpoint = {
            'iter':           itr,
            'model':          self,
            'params':         _store_params(params),
            'samples':        samples,
            'log_likelihood': self.log_likelihood(),
            'labels':         self.get_labels(),
        }
        fname = os.path.join(checkpoint_path, f'checkpoint_{itr}.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f'Checkpoint saved: {fname}')

    @classmethod
    def load_checkpoint(cls, checkpoint_file):
        """Load a checkpoint saved by fit().

        Returns:
            (model, samples, itr): fully restored model, stored snapshots,
            and the iteration number at save time.

        To resume::

            model, samples, itr = ARHMM.load_checkpoint('checkpoints/checkpoint_100.pkl')
            samples += model.fit(key, data_list, n_iter=100)
        """
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        return checkpoint['model'], checkpoint['samples'], checkpoint['iter']


    def viterbi(self, params, data_list, group_ids=None):
        """MAP state sequences via Viterbi decoding.

        Args:
            params:    Parameter dict (e.g. samples[-1] from fit()).
            data_list: List of (T_i, obs_dim) arrays (raw).
            group_ids: Group labels for separate_trans; falls back to pooled
                       pi if not provided.

        Returns:
            List of (T_i,) integer numpy arrays.
        """
        A, Sigma = params['A'], params['Sigma']
        log_pi0  = jnp.log(jnp.array(params['beta']) + 1e-300)

        if self.whiten_mode in ('all', 'each') and self.whitening_parameters is not None:
            data_list = self._whiten_new_data(data_list)

        results = []
        for i, data in enumerate(data_list):
            g_id    = group_ids[i] if group_ids is not None else None
            log_A   = _get_log_A(params, g_id, self.separate_trans)

            phi, x  = make_ar_features(
                np.array(data, dtype=np.float32), self.lag, affine=self.affine
            )
            if self.robust:
                log_liks = ar_log_likelihoods_student_t(
                    jnp.array(phi), jnp.array(x),
                    jnp.array(A), jnp.array(Sigma), self.nu
                )
            else:
                log_liks = ar_log_likelihoods(
                    jnp.array(phi), jnp.array(x),
                    jnp.array(A), jnp.array(Sigma)
                )
            states = hmm_viterbi(log_pi0, log_A, log_liks)
            results.append(np.array(states))
        return results

    def _E_step(self, params=None, data_list=None, group_ids=None):
        """Compute posterior state marginals p(z_t=k | x_{1:T}) for every session.

        Populates self.expected_states and returns the list.

        Args:
            params:    Parameter dict.  None → use self._params.
            data_list: List of raw arrays.  None → use cached prepped data.
            group_ids: Group labels.  None → use self._group_ids.

        Returns:
            List of (T_i, K) numpy arrays.
        """
        if params is None:
            if self._params is None:
                raise RuntimeError("No params — run fit() first.")
            params = self._params

        group_ids = group_ids if group_ids is not None else self._group_ids

        if data_list is None:
            if self._prepped_data is None:
                raise RuntimeError("No data — pass data_list explicitly.")
            prepped = self._prepped_data
        else:
            if self.whiten_mode in ('all', 'each') and self.whitening_parameters is not None:
                data_list = self._whiten_new_data(data_list)
            prepped = _prep_data(data_list, self.lag, self.affine)

        A, Sigma = params['A'], params['Sigma']
        log_pi0  = jnp.log(jnp.array(params['beta']) + 1e-300)

        results = []
        for i, (phi, x) in enumerate(prepped):
            g_id    = group_ids[i] if group_ids is not None else None
            log_A   = _get_log_A(params, g_id, self.separate_trans)

            if self.robust:
                log_liks = ar_log_likelihoods_student_t(
                    jnp.array(phi), jnp.array(x),
                    jnp.array(A), jnp.array(Sigma), self.nu
                )
            else:
                log_liks = ar_log_likelihoods(
                    jnp.array(phi), jnp.array(x),
                    jnp.array(A), jnp.array(Sigma)
                )
            gamma = hmm_expected_states(log_pi0, log_A, log_liks)
            results.append(np.array(gamma))

        self.expected_states = results
        return results

    def log_likelihood(self, params=None, data_list=None, group_ids=None):
        """Total log marginal likelihood under current parameters.

        Can be called with no arguments after fit() to use cached params and data.

        Args:
            params:    Parameter dict.  None → use self._params (requires fit()).
            data_list: List of raw (T_i, D) arrays.  None → use cached prepped data.
            group_ids: Group labels for separate_trans routing.  None → use stored.

        Returns:
            Scalar float (sum over all sessions).
        """
        if params is None:
            if self._params is None:
                raise RuntimeError("No params — run fit() first.")
            params = self._params

        group_ids_eff = group_ids if group_ids is not None else self._group_ids

        A, Sigma = params['A'], params['Sigma']
        log_pi0  = jnp.log(jnp.array(params['beta']) + 1e-300)

        if data_list is None:
            # Use cached prepped (phi, x) pairs — already whitened & lag-expanded.
            if self._prepped_data is None:
                raise RuntimeError("No data — pass data_list or run fit() first.")
            prepped      = self._prepped_data
            use_prepped  = True
        else:
            if self.whiten_mode in ('all', 'each') and self.whitening_parameters is not None:
                data_list = self._whiten_new_data(data_list)
            prepped     = _prep_data(data_list, self.lag, self.affine)
            use_prepped = False

        total = 0.0
        for i, (phi, x) in enumerate(prepped):
            g_id  = group_ids_eff[i] if group_ids_eff is not None else None
            log_A = _get_log_A(params, g_id, self.separate_trans)

            if self.robust:
                log_liks = ar_log_likelihoods_student_t(
                    jnp.array(phi), jnp.array(x),
                    jnp.array(A), jnp.array(Sigma), self.nu
                )
            else:
                log_liks = ar_log_likelihoods(
                    jnp.array(phi), jnp.array(x),
                    jnp.array(A), jnp.array(Sigma)
                )
            log_alphas = hmm_forward(log_pi0, log_A, log_liks)
            total += float(jax.nn.logsumexp(log_alphas[-1]))
        return total

    # ------------------------------------------------------------------
    # Convenience API
    # ------------------------------------------------------------------

    @property
    def num_states(self):
        """Number of discrete states K."""
        return self.K

    @property
    def stateseqs(self):
        """Most recent per-session state sequences from the Gibbs sampler.

        Each element is a (T_i - lag,) int32 array.  Use get_labels() to
        get sequences padded to the original T_i length.
        """
        return self.state_seqs

    def get_labels(self, sentinel=-5):
        """State sequences padded with a lag-length sentinel prefix.

        Prepends ``self.lag`` frames of ``sentinel`` to each sequence in
        ``self.state_seqs`` so that the labels align frame-for-frame with
        the original T-frame data arrays (matching moseq2-model's convention
        of returning -5 for the first ``nlags`` frames).

        Args:
            sentinel: Integer fill value for the prefix frames (default -5).

        Returns:
            List of (T_i,) int32 numpy arrays.
        """
        if self.state_seqs is None:
            raise RuntimeError("No state sequences — run fit() first.")
        return [
            np.concatenate([
                np.full(self.lag, sentinel, dtype=np.int32),
                np.asarray(s, dtype=np.int32),
            ])
            for s in self.state_seqs
        ]

    def heldout_viterbi(self, data, group_id=None):
        """Viterbi decode a single heldout session using cached parameters.

        Convenience wrapper around viterbi() for single-session inference
        after fit().  Applies the stored whitening transform if whiten=True.

        Args:
            data:     (T, obs_dim) array.
            group_id: Group label for separate_trans routing (None → pooled pi).

        Returns:
            (T - lag,) int32 numpy array of MAP state assignments.
        """
        if self._params is None:
            raise RuntimeError("No params — run fit() first.")
        gids = [group_id] if group_id is not None else None
        return self.viterbi(self._params, [data], group_ids=gids)[0]


# ---------------------------------------------------------------------------
# AR-HSMM
# ---------------------------------------------------------------------------

class ARHSMM(ARHMM):
    """Autoregressive HSMM — extends ARHMM with explicit Poisson durations.

    Self-transitions are zeroed so the duration distribution governs dwell
    times instead of kappa.  Inherits affine / whiten / empirical_bayes from
    ARHMM.  separate_trans and robust are not yet supported on ARHSMM.

    Extra Parameters
    ----------------
    max_dur : int        Maximum duration considered.
    expected_dur : float Prior expected duration in frames.
    """

    def __init__(
        self,
        n_states: int         = 100,
        obs_dim: int          = 10,
        ar_lags: int          = 3,
        alpha: float          = 5.7,
        kappa: float          = 1e6,
        gamma: float          = 999.0,
        affine: bool          = True,
        whiten: str           = 'all',
        empirical_bayes: bool = True,
        max_dur: int          = 100,
        expected_dur: float   = 20.0,
        seed: int             = 0,
    ):
        super().__init__(
            n_states, obs_dim, ar_lags, alpha, kappa, gamma,
            affine, whiten, empirical_bayes,
            separate_trans=False, robust=False, seed=seed,
        )
        self.max_dur   = max_dur
        self.dur_prior = default_duration_prior(expected_dur)

    def _gibbs_step(self, key, params, prepped_data, group_ids=None):
        A, Sigma = params['A'], params['Sigma']
        beta     = params['beta']
        lam      = params['lam']

        pi_ns = np.array(params['pi'])
        np.fill_diagonal(pi_ns, 0.0)
        row_sums = pi_ns.sum(axis=1, keepdims=True)
        pi_ns /= np.where(row_sums == 0, 1.0, row_sums)

        log_A   = jnp.log(jnp.array(pi_ns) + 1e-300)
        log_pi0 = jnp.log(jnp.array(beta)  + 1e-300)
        log_dur = poisson_log_dur_matrix(
            jnp.array(lam, dtype=jnp.float32), self.max_dur
        )

        state_seqs = []
        all_phi    = []
        all_x      = []

        for phi, x in prepped_data:
            phi_j    = jnp.array(phi)
            x_j      = jnp.array(x)
            log_liks = ar_log_likelihoods(phi_j, x_j, jnp.array(A), jnp.array(Sigma))

            log_F = hsmm_forward(log_pi0, log_A, log_dur, log_liks, self.max_dur)
            key, subkey = jax.random.split(key)
            states = hsmm_backward_sample(
                subkey, np.array(log_A), np.array(log_dur),
                log_F, np.array(log_liks), self.max_dur
            )
            state_seqs.append(states)
            all_phi.append(phi_j)
            all_x.append(x_j)

        phi_cat    = jnp.concatenate(all_phi,  axis=0)
        x_cat      = jnp.concatenate(all_x,    axis=0)
        states_cat = jnp.array(np.concatenate(state_seqs, axis=0))
        stats      = compute_sufficient_stats(phi_cat, x_cat, states_cat, self.K)
        A_new, Sigma_new = sample_obs_params(stats, self.obs_prior, self.rng)

        pi_new, beta_new = sample_transitions(
            self.rng, state_seqs, beta, self.trans_params
        )
        lam_new = sample_duration_params(
            self.rng, state_seqs, lam, self.dur_prior, self.K
        )

        new_params = dict(
            A=A_new, Sigma=Sigma_new, pi=pi_new, beta=beta_new, lam=lam_new
        )
        self.state_seqs = state_seqs
        self._obs_stats = stats
        self._params    = new_params

        return new_params, state_seqs

    def init_params(self, key, data_list, group_ids=None):
        params = super().init_params(key, data_list, group_ids=group_ids)
        a_0, b_0 = self.dur_prior['a_0'], self.dur_prior['b_0']
        params['lam'] = self.rng.gamma(a_0, 1.0 / b_0, size=self.K)
        return params

    def viterbi(self, params, data_list, group_ids=None):
        """Approximate MAP state sequence for HSMM via forward-message argmax.

        .. warning::
            This is a **filtered estimate**, not a true MAP sequence.
            ``states[t] = argmax_k log_F[t, k]`` selects the most likely state
            at each time step given *all past observations*, but it does not
            account for future observations or the duration model during the
            backward pass.  A proper HSMM MAP decode requires an HSMM Viterbi
            algorithm (O(T * D * K²)) which is not yet implemented.

            For sampling-based MAP use the posterior mode of the Gibbs chain
            (e.g. the last sample) and call ``model.state_seqs`` directly.
        """
        A, Sigma = params['A'], params['Sigma']
        lam      = params['lam']

        pi_ns = np.array(params['pi'])
        np.fill_diagonal(pi_ns, 0.0)
        row_sums = pi_ns.sum(axis=1, keepdims=True)
        pi_ns /= np.where(row_sums == 0, 1.0, row_sums)

        log_A   = jnp.log(jnp.array(pi_ns) + 1e-300)
        log_pi0 = jnp.log(jnp.array(params['beta']) + 1e-300)
        log_dur = poisson_log_dur_matrix(
            jnp.array(lam, dtype=jnp.float32), self.max_dur
        )

        if self.whiten_mode in ('all', 'each') and self.whitening_parameters is not None:
            data_list = self._whiten_new_data(data_list)

        results = []
        for data in data_list:
            phi, x   = make_ar_features(
                np.array(data, dtype=np.float32), self.lag, affine=self.affine
            )
            log_liks = ar_log_likelihoods(
                jnp.array(phi), jnp.array(x),
                jnp.array(A), jnp.array(Sigma)
            )
            log_F  = hsmm_forward(log_pi0, log_A, log_dur, log_liks, self.max_dur)
            # Filtered-estimate approximation: argmax over states at each frame.
            # NOTE: not the true MAP sequence; see docstring above.
            states = np.argmax(np.array(log_F), axis=1)
            results.append(states)
        return results

    def log_likelihood(self, params, data_list, group_ids=None):
        A, Sigma = params['A'], params['Sigma']
        lam      = params['lam']

        pi_ns = np.array(params['pi'])
        np.fill_diagonal(pi_ns, 0.0)
        row_sums = pi_ns.sum(axis=1, keepdims=True)
        pi_ns /= np.where(row_sums == 0, 1.0, row_sums)

        log_A   = jnp.log(jnp.array(pi_ns) + 1e-300)
        log_pi0 = jnp.log(jnp.array(params['beta']) + 1e-300)
        log_dur = poisson_log_dur_matrix(
            jnp.array(lam, dtype=jnp.float32), self.max_dur
        )

        if self.whiten_mode in ('all', 'each') and self.whitening_parameters is not None:
            data_list = self._whiten_new_data(data_list)

        total = 0.0
        for data in data_list:
            phi, x   = make_ar_features(
                np.array(data, dtype=np.float32), self.lag, affine=self.affine
            )
            log_liks = ar_log_likelihoods(
                jnp.array(phi), jnp.array(x),
                jnp.array(A), jnp.array(Sigma)
            )
            log_F  = hsmm_forward(log_pi0, log_A, log_dur, log_liks, self.max_dur)
            total += float(jax.nn.logsumexp(log_F[-1]))
        return total
