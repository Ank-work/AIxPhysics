"""AI: learn physical parameters from noisy observations.

This is a classic "AI for science" inverse problem: given noisy measurements of
an oscillator's position over time, recover the hidden physical parameters
(``omega0`` and ``gamma``) that generated the data. We do this by minimizing the
mean-squared residual between the simulated trajectory and the observations with
a bounded least-squares optimizer -- a physics-informed parameter fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .simulation import simulate_damped_oscillator


@dataclass(frozen=True)
class LearnedParameters:
    """Result of a parameter-learning run.

    Attributes:
        omega0: Recovered natural angular frequency (rad/s).
        gamma: Recovered damping coefficient (1/s).
        rmse: Root-mean-squared error between the fitted model and the data.
        n_iterations: Number of optimizer function evaluations.
        success: Whether the optimizer reported convergence.
    """

    omega0: float
    gamma: float
    rmse: float
    n_iterations: int
    success: bool


def add_observation_noise(
    x: np.ndarray, noise_std: float, seed: int | None = None
) -> np.ndarray:
    """Return ``x`` corrupted with zero-mean Gaussian measurement noise."""
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, noise_std, size=x.shape)


def learn_parameters(
    t: np.ndarray,
    x_observed: np.ndarray,
    x0: float,
    v0: float,
    initial_guess: tuple[float, float] = (1.0, 0.1),
    bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (1e-3, 0.0),
        (50.0, 10.0),
    ),
) -> LearnedParameters:
    """Recover ``(omega0, gamma)`` from noisy position observations.

    Args:
        t: Time grid of the observations (must match a uniform simulation grid).
        x_observed: Noisy position measurements at ``t``.
        x0: Known initial position.
        v0: Known initial velocity.
        initial_guess: Starting ``(omega0, gamma)`` for the optimizer.
        bounds: ``((omega0_min, gamma_min), (omega0_max, gamma_max))``.

    Returns:
        A :class:`LearnedParameters` with the recovered values and fit quality.
    """
    t = np.asarray(t, dtype=float)
    x_observed = np.asarray(x_observed, dtype=float)
    if t.shape != x_observed.shape:
        raise ValueError("t and x_observed must have the same shape")

    t_max = float(t[-1])
    n_points = int(t.size)

    def residuals(params: np.ndarray) -> np.ndarray:
        omega0, gamma = params
        _, x_model, _ = simulate_damped_oscillator(
            omega0=omega0,
            gamma=gamma,
            x0=x0,
            v0=v0,
            t_max=t_max,
            n_points=n_points,
        )
        return x_model - x_observed

    result = least_squares(
        residuals,
        x0=np.asarray(initial_guess, dtype=float),
        bounds=(np.asarray(bounds[0]), np.asarray(bounds[1])),
        method="trf",
    )

    final_residuals = result.fun
    rmse = float(np.sqrt(np.mean(final_residuals**2)))
    return LearnedParameters(
        omega0=float(result.x[0]),
        gamma=float(result.x[1]),
        rmse=rmse,
        n_iterations=int(result.nfev),
        success=bool(result.success),
    )
