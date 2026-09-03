"""Command-line entry point tying the physics and AI pieces together.

Running the CLI performs a full end-to-end experiment:

1. Simulate a damped oscillator with known "true" parameters (physics).
2. Corrupt the trajectory with Gaussian measurement noise.
3. Learn the parameters back from the noisy data (AI / inverse problem).
4. Report how well the recovered parameters match the truth and, optionally,
   save a plot of the truth, the noisy data and the learned fit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .learning import add_observation_noise, learn_parameters
from .simulation import simulate_damped_oscillator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aixphysics",
        description="Simulate a damped oscillator and learn its parameters from noisy data.",
    )
    parser.add_argument("--omega0", type=float, default=2.0, help="True natural frequency (rad/s).")
    parser.add_argument("--gamma", type=float, default=0.3, help="True damping coefficient (1/s).")
    parser.add_argument("--x0", type=float, default=1.0, help="Initial position.")
    parser.add_argument("--v0", type=float, default=0.0, help="Initial velocity.")
    parser.add_argument("--t-max", type=float, default=10.0, help="Simulated time (s).")
    parser.add_argument("--n-points", type=int, default=400, help="Number of samples.")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Measurement noise std.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise.")
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="If set, save a PNG figure of truth vs. noisy data vs. learned fit.",
    )
    return parser


def _save_plot(
    path: Path, t, x_true, x_noisy, x_learned, true_params, learned
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(t, x_noisy, s=10, alpha=0.4, color="#888888", label="noisy observations")
    ax.plot(t, x_true, color="#1f77b4", lw=2, label=f"truth (ω₀={true_params[0]:.2f}, γ={true_params[1]:.2f})")
    ax.plot(
        t,
        x_learned,
        color="#d62728",
        lw=2,
        ls="--",
        label=f"learned (ω₀={learned.omega0:.2f}, γ={learned.gamma:.2f})",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position")
    ax.set_title("AIxPhysics: recovering oscillator parameters from noisy data")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    t, x_true, _ = simulate_damped_oscillator(
        omega0=args.omega0,
        gamma=args.gamma,
        x0=args.x0,
        v0=args.v0,
        t_max=args.t_max,
        n_points=args.n_points,
    )
    x_noisy = add_observation_noise(x_true, noise_std=args.noise_std, seed=args.seed)

    learned = learn_parameters(t, x_noisy, x0=args.x0, v0=args.v0)

    _, x_learned, _ = simulate_damped_oscillator(
        omega0=learned.omega0,
        gamma=learned.gamma,
        x0=args.x0,
        v0=args.v0,
        t_max=args.t_max,
        n_points=args.n_points,
    )

    omega0_err = abs(learned.omega0 - args.omega0)
    gamma_err = abs(learned.gamma - args.gamma)

    print("AIxPhysics end-to-end experiment")
    print("=" * 40)
    print(f"{'parameter':<10}{'true':>10}{'learned':>12}{'abs err':>12}")
    print(f"{'omega0':<10}{args.omega0:>10.4f}{learned.omega0:>12.4f}{omega0_err:>12.4f}")
    print(f"{'gamma':<10}{args.gamma:>10.4f}{learned.gamma:>12.4f}{gamma_err:>12.4f}")
    print("-" * 40)
    print(f"fit RMSE           : {learned.rmse:.5f}")
    print(f"optimizer evals    : {learned.n_iterations}")
    print(f"optimizer success  : {learned.success}")

    if args.plot is not None:
        _save_plot(
            args.plot,
            t,
            x_true,
            x_noisy,
            x_learned,
            (args.omega0, args.gamma),
            learned,
        )
        print(f"saved plot to      : {args.plot}")

    return 0 if learned.success else 1


if __name__ == "__main__":
    sys.exit(main())
