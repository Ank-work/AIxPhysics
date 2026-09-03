"""Physics: numerical simulation of a damped harmonic oscillator.

The oscillator obeys

    x'' + 2*gamma*x' + omega0**2 * x = 0

which describes systems ranging from a mass on a spring with friction to an RLC
circuit. The state ``y = [x, v]`` is integrated with an adaptive Runge-Kutta
method (``scipy.integrate.solve_ivp``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DampedOscillator:
    """Parameters describing a damped harmonic oscillator.

    Attributes:
        omega0: Natural angular frequency (rad/s), must be positive.
        gamma: Damping coefficient (1/s), must be non-negative.
    """

    omega0: float
    gamma: float

    def __post_init__(self) -> None:
        if self.omega0 <= 0:
            raise ValueError("omega0 must be positive")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")

    def derivative(self, _t: float, state: np.ndarray) -> list[float]:
        """Right-hand side of the first-order ODE system."""
        x, v = state
        return [v, -2.0 * self.gamma * v - self.omega0**2 * x]


def simulate_damped_oscillator(
    omega0: float,
    gamma: float,
    x0: float = 1.0,
    v0: float = 0.0,
    t_max: float = 10.0,
    n_points: int = 400,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a damped harmonic oscillator.

    Args:
        omega0: Natural angular frequency (rad/s).
        gamma: Damping coefficient (1/s).
        x0: Initial position.
        v0: Initial velocity.
        t_max: Total simulated time (s).
        n_points: Number of evenly spaced output samples.

    Returns:
        Tuple ``(t, x, v)`` of 1-D arrays with the time grid, position and
        velocity trajectories.
    """
    if t_max <= 0:
        raise ValueError("t_max must be positive")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    oscillator = DampedOscillator(omega0=omega0, gamma=gamma)
    t_eval = np.linspace(0.0, t_max, n_points)
    solution = solve_ivp(
        oscillator.derivative,
        t_span=(0.0, t_max),
        y0=[x0, v0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")

    t = solution.t
    x = solution.y[0]
    v = solution.y[1]
    return t, x, v


def total_energy(
    omega0: float, x: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Return the (unit-mass) mechanical energy 0.5*v**2 + 0.5*omega0**2*x**2."""
    return 0.5 * v**2 + 0.5 * omega0**2 * x**2
