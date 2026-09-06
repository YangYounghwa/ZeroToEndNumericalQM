"""PyTorch time, grid, and domain studies with sparse SciPy comparisons."""

import free_gaussian_numpy_solution as reference
import free_gaussian_torch_solution as solution
import torch
from scipy.sparse import csr_matrix


def time_step_convergence() -> None:
    """Hold H fixed: compare PyTorch CN with PyTorch exp and sparse SciPy."""
    grid, spacing = solution.make_grid(100, -12.0, 12.0)
    hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
    initial = solution.gaussian_wave_packet(
        grid, center=-2.0, width=0.7, wave_number=1.0
    )
    final_time = 0.8
    exact = solution.matrix_exponential_state(initial, hamiltonian, spacing, final_time)
    sparse_h = csr_matrix(hamiltonian.numpy())
    sparse = reference.sparse_exponential_state(
        initial.numpy(), sparse_h, spacing, final_time
    )
    print("PyTorch time-step convergence on one fixed spatial grid")
    print(
        f"Torch exp / SciPy exp-action error: {solution.state_l2_error(exact, torch.from_numpy(sparse), spacing).item():.3e}"
    )
    print("dt        state error      norm error    Torch/SciPy CN error")
    for dt in (0.08, 0.04, 0.02, 0.01):
        steps = round(final_time / dt)
        _, states = solution.propagate_crank_nicolson(
            initial, hamiltonian, spacing, dt, steps
        )
        _, scipy_states = reference.propagate_crank_nicolson(
            initial.numpy(), sparse_h, spacing, dt, steps
        )
        error = solution.state_l2_error(states[-1], exact, spacing).item()
        norm_error = abs((spacing * torch.sum(torch.abs(states[-1]) ** 2)).item() - 1)
        delta = solution.state_l2_error(
            states[-1], torch.from_numpy(scipy_states[-1]), spacing
        ).item()
        print(f"{dt:>7.3f}  {error:>14.3e}  {norm_error:>12.3e}  {delta:>20.3e}")


def grid_convergence() -> None:
    """Use matrix exponentials to isolate spatial error from CN time error."""
    final_time = 0.8
    print("\nPyTorch spatial convergence at fixed domain; no CN time step")
    print("points   spacing       continuum state error")
    for points in (79, 159, 239):
        grid, spacing = solution.make_grid(points, -12.0, 12.0)
        hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
        initial = solution.gaussian_wave_packet(
            grid, center=-2.0, width=0.7, wave_number=1.0
        )
        numerical = solution.matrix_exponential_state(
            initial, hamiltonian, spacing, final_time
        )
        exact = solution.analytical_wavefunction(
            grid, final_time, center=-2.0, width=0.7, wave_number=1.0
        )
        error = solution.state_l2_error(numerical, exact, spacing).item()
        print(f"{points:>6}  {spacing:>10.3e}  {error:>21.3e}")


def domain_convergence() -> None:
    """Enlarge the domain at fixed dx; compare with the infinite-domain state."""
    final_time = 2.0
    print("\nPyTorch domain convergence at dx = 0.1; no CN time step")
    print("half width   continuum state error    probability in edge strips")
    for half_width in (3.0, 4.0, 6.0, 8.0):
        grid, spacing = solution.make_grid(
            round(2 * half_width / 0.1) - 1, -half_width, half_width
        )
        hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
        initial = solution.gaussian_wave_packet(
            grid, center=-2.0, width=0.7, wave_number=1.0
        )
        numerical = solution.matrix_exponential_state(
            initial, hamiltonian, spacing, final_time
        )
        exact = solution.analytical_wavefunction(
            grid, final_time, center=-2.0, width=0.7, wave_number=1.0
        )
        error = solution.state_l2_error(numerical, exact, spacing).item()
        edge = torch.abs(grid) > half_width - 1.0
        edge_probability = spacing * torch.sum(torch.abs(numerical[edge]) ** 2)
        print(f"{half_width:>10.1f}  {error:>21.3e}  {edge_probability.item():>26.3e}")
    print("Edge strips are 1 unit wide; a small final edge probability alone")
    print("cannot rule out earlier reflections. Domain error plateaus at grid error.")


def main() -> None:
    time_step_convergence()
    grid_convergence()
    domain_convergence()


if __name__ == "__main__":
    main()
