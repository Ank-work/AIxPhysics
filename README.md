# AIxPhysics

AI meets physics experiments and simulations.

This repository pairs a small **physics simulator** with an **AI/learning**
routine that recovers the underlying physical parameters from noisy data. It is
a compact, fully runnable "AI for science" inverse-problem demonstration.

- **Physics** (`aixphysics/simulation.py`): numerically integrates a damped
  harmonic oscillator, `x'' + 2·γ·x' + ω₀²·x = 0`, with `scipy`.
- **AI** (`aixphysics/learning.py`): given noisy position measurements, learns
  the hidden parameters `(ω₀, γ)` by minimizing the physics-model residual
  (a bounded least-squares inverse problem).
- **CLI** (`aixphysics/cli.py`): runs the full experiment end-to-end and can
  save a plot comparing truth, noisy data, and the learned fit.

## Setup

The project targets Python 3.10+. Create a virtual environment and install:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

In Cloud Agents this is handled automatically by the `install` step in
[`.cursor/environment.json`](.cursor/environment.json).

## Run the experiment

```bash
.venv/bin/python -m aixphysics --plot outputs/experiment.png
```

Example output:

```
AIxPhysics end-to-end experiment
========================================
parameter       true     learned     abs err
omega0        2.0000      2.0001      0.0001
gamma         0.3000      0.3005      0.0005
----------------------------------------
fit RMSE           : 0.04xx
optimizer success  : True
```

Useful flags: `--omega0`, `--gamma`, `--x0`, `--v0`, `--t-max`, `--n-points`,
`--noise-std`, `--seed`, `--plot`.

## Tests

```bash
.venv/bin/pytest
```
