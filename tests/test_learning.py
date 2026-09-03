import numpy as np

from aixphysics.learning import add_observation_noise, learn_parameters
from aixphysics.simulation import simulate_damped_oscillator


def test_recovers_parameters_from_clean_data():
    true_omega0, true_gamma = 2.5, 0.35
    t, x, _ = simulate_damped_oscillator(true_omega0, true_gamma, x0=1.0, v0=0.0)
    learned = learn_parameters(t, x, x0=1.0, v0=0.0)
    assert learned.success
    assert abs(learned.omega0 - true_omega0) < 1e-2
    assert abs(learned.gamma - true_gamma) < 1e-2


def test_recovers_parameters_from_noisy_data():
    true_omega0, true_gamma = 2.0, 0.3
    t, x, _ = simulate_damped_oscillator(true_omega0, true_gamma, x0=1.0, v0=0.0)
    x_noisy = add_observation_noise(x, noise_std=0.05, seed=0)
    learned = learn_parameters(t, x_noisy, x0=1.0, v0=0.0)
    assert learned.success
    assert abs(learned.omega0 - true_omega0) < 0.1
    assert abs(learned.gamma - true_gamma) < 0.1
    assert learned.rmse < 0.1


def test_noise_is_reproducible_with_seed():
    _, x, _ = simulate_damped_oscillator(2.0, 0.3)
    a = add_observation_noise(x, noise_std=0.1, seed=123)
    b = add_observation_noise(x, noise_std=0.1, seed=123)
    assert np.array_equal(a, b)
