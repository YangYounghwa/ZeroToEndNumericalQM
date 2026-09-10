"""Separate 2D time, grid, domain, and analytical validation tables."""

import argparse
from math import pi, sqrt
from pathlib import Path

import torch
import wave_packet_2d_numpy_solution as reference
import wave_packet_2d_torch_solution as solution


def exact_coherent_state(result: solution.Evolution2D) -> torch.Tensor:
    """Continuum normalized Gaussian at the final time, up to a global phase."""
    stiffness = solution.stiffness_matrix()
    centers, momenta, _, _ = solution.coherent_reference(result.times[-1:], stiffness)
    center = (float(centers[0, 0]), float(centers[0, 1]))
    momentum = (float(momenta[0, 0]), float(momenta[0, 1]))
    frequencies, _ = solution.normal_modes(stiffness)
    normalization: float = float(frequencies.prod()) ** 0.25 / sqrt(pi)
    return normalization * solution.coherent_packet(
        result.grid, stiffness, center, momentum
    )


def spectral_time_study() -> list[str]:
    grid = solution.make_grid(12, 10, 4, 3)
    kinetic = solution.kinetic_energy(grid)
    potential = (
        0.7
        * torch.cos(torch.pi * grid.x[None, :] / 4)
        * torch.sin(torch.pi * grid.y[:, None] / 3)
    )
    initial = solution.normalize_batch(
        solution.free_gaussian(grid, center=(-1, 0), sigma=(0.8, 0.7))[:, :, None],
        grid.area,
    )
    hamiltonian = solution.spectral_hamiltonian(kinetic, potential)
    exact: torch.Tensor = (
        torch.linalg.matrix_exp(-1j * hamiltonian) @ initial.reshape(-1)
    ).reshape(10, 12)
    rows = [
        "## Same-grid spectral time-step reference",
        "",
        "Nx=12, Ny=10, Lx=4, Ly=3, T=1. V=0.7*cos(pi*x/4)*sin(pi*y/3).",
        "Compare FFT2 splitting with the exact exponential of the same 120x120 spectral matrix.",
        "",
        "| dt | Phase-aligned state error | Previous/current error |",
        "| --- | --- | --- |",
    ]
    previous: float | None = None
    for dt in (0.1, 0.05, 0.025, 0.0125):
        _, history = solution.propagate_batch(
            initial, kinetic, potential, grid.area, dt, round(1 / dt), 1000
        )
        error = solution.phase_aligned_error(history[-1, :, :, 0], exact, grid.area)
        ratio = "—" if previous is None else f"{previous / error:.3f}"
        rows.append(f"| {dt:g} | {error:.3e} | {ratio} |")
        previous = error
    return rows


def coupled_time_study() -> list[str]:
    rows = [
        "## Coupled coherent-state time convergence",
        "",
        "Default Nx=96, Ny=80, Lx=10, Ly=8, final time=6; snapshots at intervals no greater than 0.2.",
        "State, center, covariance, and energy references are the infinite-plane normal-mode solution.",
        "Errors in observables are maxima over saved times; state error is at the final time.",
        "",
        "| dt | State error | Center error | Covariance error | Energy error | Norm error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dt in (0.08, 0.04, 0.02, 0.01):
        result = solution.solve_coupled(
            time_step=dt, num_steps=round(6 / dt), store_every=max(1, int(0.2 / dt))
        )
        centers, covariance = solution.position_statistics(result)
        exact_centers, _, exact_covariance, energy = solution.coherent_reference(
            result.times, solution.stiffness_matrix()
        )
        error = solution.phase_aligned_error(
            result.wavefunctions[-1], exact_coherent_state(result), result.grid.area
        )
        norms = result.grid.area * result.wavefunctions.abs().square().sum((1, 2))
        rows.append(
            f"| {dt:g} | {error:.3e} | {float((centers - exact_centers).abs().max()):.3e} | {float((covariance - exact_covariance).abs().max()):.3e} | {float((solution.energy_expectations(result) - energy).abs().max()):.3e} | {float((norms - 1).abs().max()):.3e} |"
        )
    rows += [
        "",
        "Conserved norm does not make the trajectory exact. Reducing dt improves the physical observables as well.",
    ]
    return rows


def grid_study() -> list[str]:
    rows = [
        "## Spatial-grid convergence",
        "",
        "Keep Lx=10, Ly=8, dt=0.005, final time=6 fixed. Reference grid: Nx=160, Ny=128.",
        "Compare the same final state on nested points without normalizing the sampled reference.",
        "",
        "| Nx | Ny | dx | dy | State error to fine grid | State error to continuum | NumPy state difference |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    fine = solution.solve_coupled(
        nx=160, ny=128, time_step=0.005, num_steps=1200, store_every=1200
    )
    for nx, ny in ((20, 16), (40, 32), (80, 64)):
        result = solution.solve_coupled(
            nx=nx, ny=ny, time_step=0.005, num_steps=1200, store_every=1200
        )
        numpy_result = reference.solve_coupled(
            nx=nx, ny=ny, time_step=0.005, num_steps=1200, store_every=1200
        )
        spatial_error = solution.phase_aligned_error(
            result.wavefunctions[-1],
            fine.wavefunctions[-1, :: 128 // ny, :: 160 // nx],
            result.grid.area,
        )
        analytical_error = solution.phase_aligned_error(
            result.wavefunctions[-1], exact_coherent_state(result), result.grid.area
        )
        library_error = solution.phase_aligned_error(
            result.wavefunctions[-1],
            torch.from_numpy(numpy_result.wavefunctions[-1]),
            result.grid.area,
        )
        rows.append(
            f"| {nx} | {ny} | {result.grid.dx:g} | {result.grid.dy:g} | {spatial_error:.3e} | {analytical_error:.3e} | {library_error:.3e} |"
        )
    rows += [
        "",
        "Once grid error is negligible, the remaining continuum error is chiefly time splitting.",
        "Refining the grid cannot remove a fixed-dt error.",
    ]
    return rows


def domain_study() -> list[str]:
    rows = [
        "## Domain convergence at fixed spacing",
        "",
        "dx=dy=0.25, dt=0.02, final time=6, unchanged coupled packet and potential.",
        "State reference: Lx=10, Ly=8 on the same spacing. Edge strips have width 1.5.",
        "Sample every 0.1 time unit; corners are counted once in the edge union.",
        "",
        "| Lx | Ly | State error to large box | Maximum center error | Maximum edge probability |",
        "| --- | --- | --- | --- | --- |",
    ]
    fine = solution.solve_coupled(nx=80, ny=64, store_every=5)
    for half_x, half_y in ((4, 3), (6, 4), (8, 6)):
        nx, ny = 8 * half_x, 8 * half_y
        result = solution.solve_coupled(nx, ny, half_x, half_y, store_every=5)
        offset_x, offset_y = 4 * (10 - half_x), 4 * (8 - half_y)
        common = fine.wavefunctions[
            -1, offset_y : offset_y + ny, offset_x : offset_x + nx
        ]
        error = solution.phase_aligned_error(
            result.wavefunctions[-1], common, result.grid.area
        )
        center, _ = solution.position_statistics(result)
        exact, _, _, _ = solution.coherent_reference(
            result.times, solution.stiffness_matrix()
        )
        rows.append(
            f"| {half_x} | {half_y} | {error:.3e} | {float((center - exact).abs().max()):.3e} | {float(solution.boundary_probability(result).max()):.3e} |"
        )
    rows += [
        "",
        "The periodic extension of a quadratic potential is not an infinite oscillator.",
        "The comparison is valid only while packet tails at the seams are negligible.",
        "Small-box errors include initial truncation and later boundary effects.",
    ]
    return rows


def free_packet_study() -> list[str]:
    grid = solution.make_grid(128, 90, 20, 18)
    kinetic = solution.kinetic_energy(grid)
    initial = solution.free_gaussian(grid)
    times, history = solution.propagate_batch(
        initial[:, :, None], kinetic, torch.zeros_like(kinetic), grid.area, 0.1, 30, 10
    )
    result = solution.Evolution2D(
        grid, kinetic, torch.zeros_like(kinetic), times, history[:, :, :, 0]
    )
    centers, covariance = solution.position_statistics(result)
    rows = [
        "## Free anisotropic Gaussian",
        "",
        "Nx=128, Ny=90, Lx=20, Ly=18, sigma=(1,1.4), center=(-3,1), momentum=(0.8,-0.4).",
        "For V=0 the kinetic step is exact on the Fourier grid; there is no splitting error.",
        "Remaining differences from the infinite plane come from sampling, periodic boundaries, and roundoff.",
        "",
        "| Time | State error to continuum | Mean x | Mean y | Variance x | Variance y |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for index, time in enumerate(times):
        error = solution.phase_aligned_error(
            result.wavefunctions[index],
            solution.free_gaussian(grid, float(time)),
            grid.area,
        )
        rows.append(
            f"| {float(time):g} | {error:.3e} | {float(centers[index, 0]):.9f} | {float(centers[index, 1]):.9f} | {float(covariance[index, 0, 0]):.9f} | {float(covariance[index, 1, 1]):.9f} |"
        )
    return rows


def write_report(output: Path) -> None:
    rows = [
        "# Two-dimensional wave-packet validation",
        "",
        f"Generated with PyTorch {torch.__version__}, CPU float64/complex128, m=hbar=1.",
        "Default coupled potential: V=(x^2+0.7*x*y+1.69*y^2)/2.",
        "Default coherent packet: center=(-2,1), momentum=(0.4,-0.6), covariance from the coupled ground state.",
        "Lx and Ly denote half extents. Fields are indexed (y,x); batches add a final axis.",
        "",
    ]
    for study in (
        spectral_time_study,
        coupled_time_study,
        grid_study,
        domain_study,
        free_packet_study,
    ):
        rows.extend(study())
        rows.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "VALIDATION.md",
    )
    args = parser.parse_args()
    write_report(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
