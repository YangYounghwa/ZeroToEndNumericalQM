"""Grid convergence for the two-dimensional stationary solver."""

import stationary_2d_torch_solution as torch_solution
import torch
from stationary_2d_numpy_solution import (
    analytical_oscillator_energies,
    anisotropic_oscillator_potential,
    coupled_quartic_potential,
    solve_stationary_2d,
)


def separable_validation_convergence() -> None:
    """Validate grid convergence against an analytical spectrum."""
    exact_ground = float(analytical_oscillator_energies(1)[0])
    print("Separable oscillator validation")
    print("Nx=Ny   spacing x    numerical E0    absolute error")
    for num_points in (12, 18, 26, 36):
        result = solve_stationary_2d(
            anisotropic_oscillator_potential,
            num_x=num_points,
            num_y=num_points,
            num_states=1,
            x_min=-7.0,
            x_max=7.0,
            y_min=-7.0,
            y_max=7.0,
        )
        error = abs(float(result.energies[0]) - exact_ground)
        print(
            f"{num_points:>5}   {result.spacing_x:>10.3e}  "
            f"{result.energies[0]:>14.8f}  {error:>14.3e}"
        )


def nonseparable_grid_convergence() -> None:
    """Compare the coupled problem with a refined numerical reference."""
    reference = solve_stationary_2d(
        coupled_quartic_potential,
        num_x=64,
        num_y=58,
        num_states=1,
    )
    reference_energy = float(reference.energies[0])
    print("\nNonseparable coupled-quartic convergence")
    print("Nx   Ny    numerical E0    refined-reference error")
    for num_x, num_y in ((14, 12), (20, 18), (28, 26), (40, 36)):
        result = solve_stationary_2d(
            coupled_quartic_potential,
            num_x=num_x,
            num_y=num_y,
            num_states=1,
        )
        error = abs(float(result.energies[0]) - reference_energy)
        print(f"{num_x:>2}  {num_y:>3}  {result.energies[0]:>14.8f}  {error:>23.3e}")


def torch_sparse_comparison() -> None:
    """Learn dense PyTorch on small grids and compare with sparse SciPy."""
    print("PyTorch versus SciPy for the nonseparable potential")
    print("Nx   Ny   Torch E0       SciPy delta    dense H MiB")
    for nx, ny in ((10, 9), (16, 14), (22, 20)):
        result = torch_solution.solve_stationary_2d(
            torch_solution.coupled_quartic_potential,
            num_x=nx,
            num_y=ny,
            num_states=3,
        )
        sparse = solve_stationary_2d(
            coupled_quartic_potential, num_x=nx, num_y=ny, num_states=3
        )
        delta = torch.max(
            torch.abs(result.energies - torch.from_numpy(sparse.energies))
        )
        print(
            f"{nx:>2}  {ny:>3}  {result.energies[0].item():>12.8f}  "
            f"{delta.item():>12.3e}  {8 * (nx * ny) ** 2 / 2**20:>11.3f}"
        )
    print("Dense-memory figures cover H only, excluding eigenvectors and workspace.")


def domain_convergence() -> None:
    """Use sparse matrices to enlarge both axes at fixed dx = dy = 0.25."""
    print("\nSparse 2D domain study at dx = dy = 0.25")
    print("half width     E0")
    for half_width in (2.0, 3.0, 4.0, 6.0):
        points = round(2 * half_width / 0.25) - 1
        result = solve_stationary_2d(
            coupled_quartic_potential,
            num_x=points,
            num_y=points,
            num_states=1,
            x_min=-half_width,
            x_max=half_width,
            y_min=-half_width,
            y_max=half_width,
        )
        print(f"{half_width:>10.1f}  {result.energies[0]:>12.8f}")
    print("Domain convergence can plateau while grid error remains.")


def main() -> None:
    torch_sparse_comparison()
    separable_validation_convergence()
    nonseparable_grid_convergence()
    domain_convergence()


if __name__ == "__main__":
    main()
