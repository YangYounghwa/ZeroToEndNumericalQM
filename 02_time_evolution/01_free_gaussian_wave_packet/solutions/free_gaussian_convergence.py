"""Time-step convergence for free Gaussian Crank-Nicolson propagation."""

import numpy as np
from free_gaussian_numpy_solution import (
    build_free_hamiltonian,
    gaussian_wave_packet,
    make_grid,
    matrix_exponential_state,
    normalize_wavefunction,
    propagate_crank_nicolson,
    state_l2_error,
)


def time_step_convergence() -> None:
    """Compare Crank-Nicolson with a dense matrix-exponential reference."""
    grid, spacing = make_grid(140, -16.0, 16.0)
    hamiltonian = build_free_hamiltonian(len(grid), spacing)
    initial = normalize_wavefunction(
        gaussian_wave_packet(grid, center=-4.0, width=0.9, wave_number=1.5),
        spacing,
    )
    final_time = 0.8
    reference = matrix_exponential_state(initial, hamiltonian, spacing, final_time)
    print("Crank-Nicolson time-step convergence at t = 0.8")
    print("dt        steps    state L2 error    norm error")
    for time_step in (0.08, 0.04, 0.02, 0.01):
        num_steps = round(final_time / time_step)
        _, states = propagate_crank_nicolson(
            initial, hamiltonian, spacing, time_step, num_steps
        )
        error = state_l2_error(states[-1], reference, spacing)
        norm_error = abs(spacing * np.sum(np.abs(states[-1]) ** 2) - 1.0)
        print(f"{time_step:>8.3f}  {num_steps:>5}  {error:>16.3e}  {norm_error:>10.3e}")


def main() -> None:
    time_step_convergence()


if __name__ == "__main__":
    main()
