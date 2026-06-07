# jax-hsmm

A JAX-native implementation of the Autoregressive Hidden Markov Model (AR-HMM) and Autoregressive Hidden Semi-Markov Model (AR-HSMM) for behavioral segmentation. Designed as a drop-in replacement for the [moseq2-model](https://github.com/dattalab/moseq2-model) / [pyhsmm](https://github.com/mattjj/pyhsmm) / [pybasicbayes](https://github.com/mattjj/pybasicbayes) stack, with GPU support via JAX and no Cython dependencies.

## Overview

Both models segment multivariate time series (e.g., PCA scores of animal pose) into discrete behavioral states ("syllables") using Gibbs sampling under a sticky HDP-HMM prior (Fox et al., 2008).

- **ARHMM** — implicit geometric duration distribution via kappa-inflated self-transitions
- **ARHSMM** — explicit Poisson duration distribution; controls syllable length directly

The Gibbs sweep at each iteration:
1. Sample state sequences via forward-backward message passing (JAX, GPU-compatible)
2. Sample AR parameters (A, Σ) via Matrix Normal Inverse-Wishart conjugate update
3. Sample transition parameters (π, β) via sticky HDP-HMM auxiliary variable sampler
4. *(HSMM only)* Sample Poisson duration rates λ via Gamma conjugate update

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9, JAX ≥ 0.4.0, NumPy ≥ 1.24, SciPy ≥ 1.10.

For GPU support, install the appropriate `jaxlib` build for your CUDA version before installing this package. See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Quick Start

```python
import jax
import numpy as np
from jax_hsmm import ARHMM, ARHSMM

# data_list: list of (T_i, obs_dim) float32 arrays (e.g. PCA scores)
data_list = [np.random.randn(5000, 10).astype(np.float32) for _ in range(8)]

# Fit AR-HMM (kappa defaults to total frames, matching moseq2-model)
model = ARHMM(
    n_states=100,
    obs_dim=10,
    ar_lags=3,
    affine=True,
    whiten='all',
    empirical_bayes=True,
)
key = jax.random.PRNGKey(0)
samples, iter_lls = model.fit(key, data_list, n_iter=200)

# MAP state sequences (Viterbi)
labels = model.viterbi(samples[-1], data_list)

# Posterior state marginals (E-step)
gammas = model._E_step()

# Per-frame labels padded to original length (sentinel=-5 for lag frames)
padded_labels = model.get_labels()
```

## Configuration

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `n_states` | 100 | Maximum number of syllable states |
| `obs_dim` | 10 | Observation dimensionality (number of PCs) |
| `ar_lags` | 3 | Number of autoregressive lags |
| `kappa` | `None` | Self-transition stickiness. `None` sets kappa = total training frames, matching moseq2-model's default |
| `alpha` | 5.7 | HDP row concentration |
| `gamma` | 999.0 | HDP top-level concentration |
| `affine` | `True` | Include bias column in AR features |
| `whiten` | `'all'` | Whitening mode: `'all'` (pooled), `'each'` (per-session), `'none'` |
| `empirical_bayes` | `True` | Initialize Ψ₀ from a global AR MLE fit |
| `robust` | `False` | Student-t observation noise (ARHMM only) |
| `nu` | 3.0 | Degrees of freedom for robust model |
| `separate_trans` | `False` | Per-group transition matrices (ARHMM only) |

### Separate transition matrices

For experiments with multiple groups (e.g., control vs. knockout):

```python
model = ARHMM(n_states=100, obs_dim=10, ar_lags=3, separate_trans=True)
group_ids = ['ctrl', 'ctrl', 'ctrl', 'ko', 'ko', 'ko']  # one per session
samples, iter_lls = model.fit(key, data_list, n_iter=200, group_ids=group_ids)
```

All groups share observation parameters (A, Σ) and the global HDP weights (β), but each has its own transition matrix π_g.

### Robust model

```python
model = ARHMM(n_states=100, obs_dim=10, ar_lags=3, robust=True, nu=3.0)
```

Uses Student-t observation noise via auxiliary precision weights (τ_t ~ Gamma). More tolerant of outlier frames.

### HSMM

```python
model = ARHSMM(
    n_states=100,
    obs_dim=10,
    ar_lags=3,
    max_dur=100,       # maximum syllable duration in frames
    expected_dur=20.0, # prior expected duration
)
samples, iter_lls = model.fit(key, data_list, n_iter=200)

# True MAP decode via full HSMM Viterbi (O(T·D·K²))
labels = model.viterbi(samples[-1], data_list)
```

## Checkpointing

```python
samples, iter_lls = model.fit(
    key, data_list, n_iter=1000,
    checkpoint_freq=100,
    checkpoint_path='checkpoints/',
)

# Resume from checkpoint
model, samples, itr, iter_lls = ARHMM.load_checkpoint('checkpoints/checkpoint_500.pkl')
samples_new, _ = model.fit(key, data_list, n_iter=500)
```

## Applying to new data

```python
# Viterbi decode on held-out sessions using trained parameters
new_labels = model.heldout_viterbi(new_session)           # single session
new_labels = model.viterbi(samples[-1], new_data_list)    # batch
```

## Relationship to moseq2-model

This library is a faithful JAX reimplementation of `moseq2-model`'s `FastARWeakLimitStickyHDPHMM`. The following defaults are matched exactly:

- `alpha=5.7`, `gamma=999`, `K_0_scale=10` → `V_0 = 0.1·I` (prior precision on A)
- `S_0_scale=0.01` → `Ψ_0 = 0.01·I`
- `nu_0 = D + 2`
- `M_0`: lag-1 identity block, zeros elsewhere
- `kappa = None` → total training frames (moseq2-model's `kappa=None` default)
- Empirical Bayes: global AR MLE sets Ψ₀ so that E[Σ] = Σ_mle
- Whitening: Cholesky decomposition of pooled or per-session covariance
- Labels: sentinel value `-5` for the first `nlags` frames

The main additions over moseq2-model:
- GPU-accelerated message passing via `jax.lax.scan`
- Robust Student-t observations
- True HSMM Viterbi decoding (vs. the approximate filtered argmax in pyhsmm)
- No Cython build step

## Architecture

```
jax_hsmm/
├── model.py         ARHMM and ARHSMM classes, Gibbs loop, whitening
├── messages.py      HMM/HSMM forward-backward, Viterbi (JAX + NumPy)
├── observations.py  AR log-likelihoods, MNIW conjugate sampling
├── transitions.py   Sticky HDP-HMM Gibbs update (Fox et al. 2008)
├── durations.py     Poisson duration model for HSMM
└── util.py          Numerical stability utilities
```

## References

- Fox, E., Sudderth, E., Jordan, M., & Willsky, A. (2008). *An HDP-HMM for Systems with State Persistence*. ICML.
- Johnson, M., & Willsky, A. (2013). *Bayesian Nonparametric Hidden Semi-Markov Models*. JMLR.
- Wiltschko, A. et al. (2015). *Mapping Sub-Second Structure in Mouse Behavior*. Neuron.
