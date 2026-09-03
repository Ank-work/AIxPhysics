"""AIxPhysics: AI meets physics experiments and simulations.

This package pairs a small physics simulator (a damped harmonic oscillator
integrated numerically) with a learning routine that recovers the underlying
physical parameters from noisy observations. Together they form a compact,
end-to-end "AI for science" inverse-problem demonstration.
"""

from .simulation import DampedOscillator, simulate_damped_oscillator
from .learning import LearnedParameters, learn_parameters

__all__ = [
    "DampedOscillator",
    "simulate_damped_oscillator",
    "LearnedParameters",
    "learn_parameters",
]

__version__ = "0.1.0"
