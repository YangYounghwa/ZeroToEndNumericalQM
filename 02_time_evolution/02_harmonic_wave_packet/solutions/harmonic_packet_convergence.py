"""Time-step convergence for harmonic-packet propagation."""

from harmonic_packet_numpy_solution import (
    build_hamiltonian,
    coherent_state,
    make_grid,
    matrix_exponential_state,
    propagate_crank_nicolson,
    state_l2_error,
)


def time_step_convergence() -> None:
    """Compare Crank-Nicolson with the matrix-exponential reference."""
    grid, spacing = make_grid(120, -9.0, 9.0)
    _, hamiltonian = build_hamiltonian(grid, spacing)
    initial = coherent_state(grid)
    final_time = 1.0
    reference = matrix_exponential_state(initial, hamiltonian, spacing, final_time)
    print("Harmonic-packet time-step convergence at t = 1")
    print("dt        steps    state L2 error")
    for time_step in (0.1, 0.05, 0.025, 0.0125):
        num_steps = round(final_time / time_step)
        _, states = propagate_crank_nicolson(
            initial, hamiltonian, spacing, time_step, num_steps
        )
        error = state_l2_error(states[-1], reference, spacing)
        print(f"{time_step:>8.4f}  {num_steps:>5}  {error:>16.3e}")


def main() -> None:
    time_step_convergence()


if __name__ == "__main__":
    main()
