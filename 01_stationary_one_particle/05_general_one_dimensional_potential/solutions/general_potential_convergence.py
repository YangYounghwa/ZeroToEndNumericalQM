"""PyTorch convergence with dense NumPy and sparse SciPy comparisons."""

import general_potential_numpy_solution as reference
import general_potential_torch_solution as solution
import numpy as np
import torch


def grid_convergence() -> None:
    """Compare algorithms on exactly the same grid as resolution improves."""
    exact = float(solution.analytical_harmonic_energies(1)[0])
    print("PyTorch grid convergence for the shifted harmonic oscillator")
    print("points   spacing      energy error    NumPy delta     SciPy delta")
    for num_points in (50, 100, 200, 400):
        result = solution.solve_stationary(
            solution.shifted_harmonic_potential, num_points=num_points, num_states=1
        )
        dense = reference.solve_stationary(
            reference.shifted_harmonic_potential, num_points=num_points, num_states=1
        )
        sparse = reference.solve_stationary_sparse(
            reference.shifted_harmonic_potential, num_points=num_points, num_states=1
        )
        energy = result.energies[0].item()
        print(
            f"{num_points:>6}  {result.spacing:>10.3e}  {abs(energy - exact):>12.3e}  "
            f"{abs(energy - dense.energies[0]):>12.3e}  "
            f"{abs(energy - sparse.energies[0]):>12.3e}"
        )


def domain_convergence() -> None:
    """Enlarge the box about the well center at fixed spacing."""
    exact = solution.analytical_harmonic_energies(1)[0].item()
    print("\nPyTorch domain convergence at dx = 0.05, centered on x = 0.75")
    print("half width   energy error     residual")
    for half_width in (1.0, 2.0, 3.0, 5.0):
        result = solution.solve_stationary(
            solution.shifted_harmonic_potential,
            num_points=round(2 * half_width / 0.05) - 1,
            num_states=1,
            x_min=0.75 - half_width,
            x_max=0.75 + half_width,
        )
        print(
            f"{half_width:>10.1f}  {abs(result.energies[0].item() - exact):>12.3e}  "
            f"{solution.residual_norms(result)[0].item():>12.3e}"
        )
    print("A small residual does not remove finite-domain or grid error.")


def potential_reuse_study() -> None:
    """Use batched PyTorch potentials and compare each with a sparse solve."""
    grid, spacing = solution.make_grid(160)
    potentials = torch.stack((torch.zeros_like(grid), 0.5 * grid**2, 0.25 * grid**4))
    batch = solution.solve_potential_batch(grid, spacing, potentials, num_states=2)
    print("\nOne PyTorch batch, three potentials")
    print("potential          E0            E1     max SciPy delta")
    for index, name in enumerate(("free box", "harmonic", "quartic")):
        sampled = potentials[index].numpy()

        def potential(
            x: reference.FloatArray, values: reference.FloatArray = sampled
        ) -> reference.FloatArray:
            if not np.allclose(x, grid.numpy(), atol=1e-14, rtol=0):
                raise ValueError("comparison requires identical grids")
            return values

        sparse = reference.solve_stationary_sparse(
            potential, num_points=160, num_states=2
        )
        delta = np.max(np.abs(batch.energies[index].numpy() - sparse.energies))
        print(
            f"{name:<10}  {batch.energies[index, 0].item():>12.8f}  "
            f"{batch.energies[index, 1].item():>12.8f}  {delta:>14.3e}"
        )


def main() -> None:
    grid_convergence()
    domain_convergence()
    potential_reuse_study()


if __name__ == "__main__":
    main()
