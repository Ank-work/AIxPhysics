import numpy as np
import pytest

from aixphysics.simulation import (
    DampedOscillator,
    simulate_damped_oscillator,
    total_energy,
)


def test_undamped_conserves_energy():
    omega0 = 2.0
    t, x, v = simulate_damped_oscillator(omega0=omega0, gamma=0.0, x0=1.0, v0=0.0)
    energy = total_energy(omega0, x, v)
    assert np.allclose(energy, energy[0], rtol=1e-4, atol=1e-4)


def test_undamped_matches_analytic_solution():
    omega0 = 3.0
    t, x, _ = simulate_damped_oscillator(
        omega0=omega0, gamma=0.0, x0=1.0, v0=0.0, t_max=5.0, n_points=200
    )
    analytic = np.cos(omega0 * t)
    assert np.max(np.abs(x - analytic)) < 1e-4


def test_damping_reduces_energy_monotonically():
    omega0 = 2.0
    t, x, v = simulate_damped_oscillator(omega0=omega0, gamma=0.4, x0=1.0, v0=0.0)
    energy = total_energy(omega0, x, v)
    assert energy[-1] < energy[0]
    assert energy[-1] < 0.1 * energy[0]


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        DampedOscillator(omega0=-1.0, gamma=0.1)
    with pytest.raises(ValueError):
        DampedOscillator(omega0=1.0, gamma=-0.1)
    with pytest.raises(ValueError):
        simulate_damped_oscillator(omega0=1.0, gamma=0.1, n_points=1)
