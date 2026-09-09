"""Write reproducible scattering convergence tables to a Markdown report."""

import argparse
from pathlib import Path

import barrier_scattering_numpy_solution as reference
import barrier_scattering_torch_solution as solution
import numpy as np
import torch
from scipy.sparse import csr_matrix


def state_error(first: torch.Tensor, second: torch.Tensor, spacing: float) -> float:
    """Weighted L2 error on a common grid after global-phase alignment."""
    overlap = spacing * torch.vdot(second, first)
    aligned = first * torch.exp(-1j * torch.angle(overlap))
    return float(torch.sqrt(spacing * torch.sum(torch.abs(aligned - second) ** 2)))


def time_step_study() -> list[str]:
    """Hold one spatial Hamiltonian fixed and compare CN with exponentials."""
    grid, spacing, _, hamiltonian = solution.build_hamiltonian(159, 12.0)
    initial = solution.gaussian_packet(grid, center=-6.0, sigma=1.0)
    initial = solution.normalize_columns(initial[:, None], spacing)[:, 0]
    exact: torch.Tensor = torch.linalg.matrix_exp(-4j * hamiltonian) @ initial
    scipy_exact = reference.exponential_state(
        initial.numpy(), csr_matrix(hamiltonian.numpy()), spacing, 4.0
    )
    rows = [
        "## Time-step error",
        "",
        "PyTorch CN at N=159, walls ±12, x0=-6, sigma=1, k0=2, V0=2.5, a=2, T=4.",
        "The spatial Hamiltonian is identical for all time steps and references.",
        f"PyTorch/SciPy exponential state difference: {state_error(exact, torch.from_numpy(scipy_exact), spacing):.3e}.",
        "",
        "| dt | Phase-aligned state error | Previous/current error | Norm error |",
        "| --- | --- | --- | --- |",
    ]
    previous: float | None = None
    for dt in (0.1, 0.05, 0.025, 0.0125):
        steps = round(4 / dt)
        _, history = solution.propagate_batch(
            initial[:, None], hamiltonian, spacing, dt, steps, steps
        )
        final = history[-1, :, 0]
        error = state_error(final, exact, spacing)
        ratio = "—" if previous is None else f"{previous / error:.3f}"
        norm_error = abs(float(spacing * torch.sum(torch.abs(final) ** 2)) - 1)
        rows.append(f"| {dt:g} | {error:.3e} | {ratio} | {norm_error:.3e} |")
        previous = error
    return rows


def grid_study() -> list[str]:
    """Keep physical walls and midpoint-aligned barrier edges fixed."""
    exact = solution.packet_transmission_reference().item()
    quadrature_fine = solution.packet_transmission_reference(num_samples=8001).item()
    rows = [
        "## Grid error and library comparison",
        "",
        "Walls ±40, x0=-12, sigma=2, k0=2, V0=2.5, a=2, T=14, dt=0.01.",
        "At N=439, 839, 1639, dx=2/11, 2/21, 2/41 respectively.",
        "Both barrier edges remain halfway between neighboring grid points.",
        f"Infinite-domain packet reference: T={exact:.9f}; doubling quadrature samples changes it by {abs(exact - quadrature_fine):.3e}.",
        "",
        "| N | dx | Sparse T | Error to continuum packet T | Final near probability | PyTorch/SciPy T difference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for points in (439, 839, 1639):
        sparse = reference.solve_scattering(
            num_points=points, time_step=0.01, num_steps=1400, store_every=100
        )
        regions = reference.region_probabilities(sparse)
        transmission = float(regions.right[-1])
        delta = "sparse only"
        if points <= 839:
            dense = solution.solve_scattering(
                num_points=points, time_step=0.01, num_steps=1400, store_every=100
            )
            torch_t = solution.region_probabilities(dense).right[-1].item()
            delta = f"{abs(torch_t - transmission):.3e}"
        rows.append(
            f"| {points} | {sparse.spacing:.6f} | {transmission:.9f} | {abs(transmission - exact):.3e} | {regions.near[-1]:.3e} | {delta} |"
        )
    rows.extend(
        [
            "",
            "The remaining difference includes grid, finite-time, domain, and time-step error. A finite-grid result is not an exact continuum reference.",
        ]
    )
    return rows


def measurement_time_study() -> list[str]:
    """Follow one PyTorch trajectory through collision and outgoing separation."""
    result = solution.solve_scattering(num_steps=800, store_every=50)
    regions = solution.region_probabilities(result)
    edge = solution.boundary_probability(result)
    rows = [
        "## Measurement window",
        "",
        "PyTorch default grid N=439, walls ±40, dt=0.02; snapshots every 1 time unit.",
        "The near region is [-2, 2]; edge strips are 3 units wide.",
        "",
        "| Time | Left | Near | Right | Maximum saved edge probability so far |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index in (0, 6, 8, 10, 12, 14, 16):
        rows.append(
            f"| {result.times[index].item():.1f} | {regions.left[index].item():.7f} | {regions.near[index].item():.3e} | {regions.right[index].item():.7f} | {edge[: index + 1].max().item():.3e} |"
        )
    energy = solution.energy_expectations(result)
    norms = regions.left + regions.near + regions.right
    rows.extend(
        [
            "",
            f"Maximum norm error: {torch.max(torch.abs(norms - 1)).item():.3e}; maximum energy drift: {torch.max(torch.abs(energy - energy[0])).item():.3e}.",
            "Early left probability contains the incident packet. Late left/right values approximate R/T only after near probability is small and before wall returns.",
            "Saved edge diagnostics can miss events between snapshots. Check snapshot frequency and compare domains as well.",
        ]
    )
    return rows


def domain_study() -> list[str]:
    """Use sparse solves at identical spacing and compare common-grid final states."""
    large = reference.solve_scattering(
        num_points=1049, half_extent=50, time_step=0.01, num_steps=1400, store_every=10
    )
    rows = [
        "## Domain error",
        "",
        "SciPy comparison at fixed dx=2/21, dt=0.01, T=14 and unchanged packet/barrier.",
        "State errors use the common grid against walls ±50, without renormalizing the cropped reference.",
        "Edge strips are 3 units wide and sampled every 0.1 time unit.",
        "",
        "| Half extent | Right probability | State error to larger box | Maximum saved edge probability |",
        "| --- | --- | --- | --- |",
    ]
    for extent in (20, 30, 40):
        points = 21 * extent - 1
        result = reference.solve_scattering(
            num_points=points,
            half_extent=extent,
            time_step=0.01,
            num_steps=1400,
            store_every=10,
        )
        start = (large.grid.size - result.grid.size) // 2
        np.testing.assert_allclose(
            result.grid, large.grid[start : start + points], atol=1e-12, rtol=0
        )
        error = state_error(
            torch.from_numpy(result.wavefunctions[-1]),
            torch.from_numpy(large.wavefunctions[-1, start : start + points]),
            result.spacing,
        )
        transmission = reference.region_probabilities(result).right[-1]
        rows.append(
            f"| {extent} | {transmission:.9f} | {error:.3e} | {reference.boundary_probability(result).max():.3e} |"
        )
    rows.extend(
        [
            "",
            "Right-side probability can appear converged while a reflected packet is already distorted by the left wall; inspect state error too.",
        ]
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "VALIDATION.md",
    )
    arguments = parser.parse_args()
    output: Path = arguments.output
    rows = [
        "# Barrier-scattering numerical validation",
        "",
        f"Generated by barrier_scattering_convergence.py with PyTorch {torch.__version__}, CPU float64/complex128.",
        "All calculations use m=hbar=1. These are reproducible numerical experiments, not performance benchmarks.",
        "",
    ]
    for study in (time_step_study, grid_study, measurement_time_study, domain_study):
        rows.extend(study())
        rows.append("")
    output.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
