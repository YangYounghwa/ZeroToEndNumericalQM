"""Write separate tunnelling convergence experiments to Markdown."""

import argparse
from pathlib import Path

import numpy as np
import torch
import tunnelling_numpy_solution as reference
import tunnelling_torch_solution as solution


def time_step_study() -> list[str]:
    """Compare splitting with the exact exponential of the same spectral matrix."""
    x, dx, k = solution.make_grid(96, 12)
    potential = solution.gaussian_barrier(x)
    initial = solution.normalize_columns(
        solution.gaussian_packet(x, -4, 0.8)[:, None], dx
    )
    hamiltonian = solution.spectral_hamiltonian(k**2 / 2, potential)
    exact: torch.Tensor = torch.linalg.matrix_exp(-4j * hamiltonian) @ initial[:, 0]
    rows = [
        "## Time-step error on one spectral grid",
        "",
        "N=96, L=12, x0=-4, sigma=0.8, k0=1.5, final time=4. The reference is",
        "the PyTorch matrix exponential of the same spectral Hamiltonian, not a finite-difference matrix.",
        "",
        "| dt | Phase-aligned state error | Previous/current error |",
        "| --- | --- | --- |",
    ]
    previous: float | None = None
    for dt in (0.1, 0.05, 0.025, 0.0125):
        _, states = solution.propagate_batch(
            initial, k**2 / 2, potential, dx, dt, round(4 / dt), 1000
        )
        error = solution.phase_aligned_error(states[-1, :, 0], exact, dx)
        ratio = "—" if previous is None else f"{previous / error:.3f}"
        rows.append(f"| {dt:g} | {error:.3e} | {ratio} |")
        previous = error
    return rows


def scattering_time_study() -> list[str]:
    rows = [
        "## Scattering time-step and energy checks",
        "",
        "Default physical setup and N=1024; final time=30, snapshots at intervals no greater than 0.5.",
        "State reference uses dt=0.0025 on the same grid. This is a numerical reference.",
        "",
        "| dt | Right probability | State error | Maximum energy drift | Maximum norm error |",
        "| --- | --- | --- | --- | --- |",
    ]
    fine = solution.solve_tunnelling(
        time_step=0.0025, num_steps=12000, store_every=12000
    )
    for dt in (0.08, 0.04, 0.02, 0.01):
        # Use evenly spaced snapshots no farther than 0.5 apart.
        result = solution.solve_tunnelling(
            time_step=dt, num_steps=round(30 / dt), store_every=max(1, int(0.5 / dt))
        )
        regions = solution.region_probabilities(result)
        energies = solution.energy_expectations(result)
        error = solution.phase_aligned_error(
            result.wavefunctions[-1], fine.wavefunctions[-1], result.spacing
        )
        rows.append(
            f"| {dt:g} | {float(regions[-1, 2]):.9f} | {error:.3e} | {float((energies - energies[0]).abs().max()):.3e} | {float((regions.sum(1) - 1).abs().max()):.3e} |"
        )
    rows += [
        "",
        "Splitting preserves norm but not the energy of H exactly. Both state error and energy drift decrease with dt.",
    ]
    return rows


def grid_study() -> list[str]:
    rows = [
        "## Spatial grid and library comparison",
        "",
        "L=64, final time=30, dt=0.01; FFT state reference N=4096 at the same dt.",
        "Coarse states are compared with samples of the fine state without renormalizing the reference.",
        "",
        "| N | dx | FFT right probability | State error to fine grid | Maximum high-k mass | NumPy state difference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    fine = solution.solve_tunnelling(
        num_points=4096, time_step=0.01, num_steps=3000, store_every=3000
    )
    for points in (128, 256, 512, 1024):
        result = solution.solve_tunnelling(
            num_points=points, time_step=0.01, num_steps=3000, store_every=50
        )
        numpy_result = reference.solve_tunnelling(
            num_points=points, time_step=0.01, num_steps=3000, store_every=3000
        )
        error = solution.phase_aligned_error(
            result.wavefunctions[-1],
            fine.wavefunctions[-1, :: 4096 // points],
            result.spacing,
        )
        library = solution.phase_aligned_error(
            result.wavefunctions[-1],
            torch.from_numpy(numpy_result.wavefunctions[-1]),
            result.spacing,
        )
        rows.append(
            f"| {points} | {result.spacing:g} | {float(solution.region_probabilities(result)[-1, 2]):.9f} | {error:.3e} | {float(solution.high_frequency_probability(result).max()):.3e} | {library:.3e} |"
        )
    rows += [
        "",
        "### Finite-difference Crank-Nicolson comparison",
        "",
        "The Gaussian potential, box, periodic boundaries, and time are unchanged.",
        "The kinetic discretization differs; agreement is expected after grid refinement.",
        "CN uses dt=0.01; the last row is repeated at dt=0.005 to expose time error.",
        "",
        "| N | CN dt | CN right probability | Difference from fine FFT right probability |",
        "| --- | --- | --- | --- |",
    ]
    fft_transmission = float(solution.region_probabilities(fine)[-1, 2])
    for points, dt in (
        (512, 0.01),
        (1024, 0.01),
        (2048, 0.01),
        (4096, 0.01),
        (4096, 0.005),
    ):
        x, dx, _ = reference.make_grid(points, 64)
        initial = reference.gaussian_packet(x)[:, None]
        hamiltonian = reference.periodic_fd_hamiltonian(
            reference.gaussian_barrier(x), dx
        )
        steps = round(30 / dt)
        _, history = reference.propagate_cn(initial, hamiltonian, dx, dt, steps, steps)
        transmission = float(dx * np.sum(np.abs(history[-1, x > 4, 0]) ** 2))
        rows.append(
            f"| {points} | {dt:g} | {transmission:.9f} | {abs(transmission - fft_transmission):.3e} |"
        )
    rows += [
        "",
        "The fine FFT value is not an exact continuum transmission; finite time, dt, and box errors remain.",
    ]
    return rows


def domain_study() -> list[str]:
    rows = [
        "## Periodic-domain error",
        "",
        "Fixed dx=0.125, dt=0.02, final time=30, unchanged packet and barrier.",
        "Compare states on the common grid with L=80, without renormalizing the cropped reference.",
        "Boundary strips are 6 units wide; sample every 0.1 time unit.",
        "",
        "| L | Right probability | State error to L=80 | Maximum saved edge probability |",
        "| --- | --- | --- | --- |",
    ]
    fine = solution.solve_tunnelling(num_points=1280, half_extent=80, store_every=5)
    for extent in (32, 40, 48, 64):
        result = solution.solve_tunnelling(
            num_points=16 * extent, half_extent=extent, store_every=5
        )
        offset = round((80 - extent) / result.spacing)
        common = fine.wavefunctions[-1, offset : offset + len(result.grid)]
        error = solution.phase_aligned_error(
            result.wavefunctions[-1], common, result.spacing
        )
        rows.append(
            f"| {extent} | {float(solution.region_probabilities(result)[-1, 2]):.9f} | {error:.3e} | {float(solution.boundary_probability(result).max()):.3e} |"
        )
    rows += [
        "",
        "A packet leaving one edge re-enters through the other. Norm remains one even when scattering is contaminated.",
        "Edge snapshots are a warning signal, not a proof: they can miss motion between saved times.",
    ]
    return rows


def packet_width_study() -> list[str]:
    rows = [
        "## Control the incident energy spread",
        "",
        "Batched PyTorch run: L=160, N=2560, x0=-32, k0=1.5, dt=0.02, final time=44.",
        "Only sigma changes. Delta k=1/(2 sigma), mean free energy=k0^2/2+1/(8 sigma^2).",
        "P(E>V0) includes both signs of momentum. These are infinite-line incoming Gaussian estimates.",
        "",
        "| sigma | Delta k | Mean E | P(E>V0) | Final right | Final near | Maximum edge mass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    x, dx, k = solution.make_grid(2560, 160)
    potential = solution.gaussian_barrier(x)
    widths = (1.0, 2.0, 3.0, 4.0)
    initial = torch.stack(
        [solution.gaussian_packet(x, -32, sigma) for sigma in widths], dim=1
    )
    times, history = solution.propagate_batch(
        initial, k**2 / 2, potential, dx, 0.02, 2200, 25
    )
    for column, sigma in enumerate(widths):
        result = solution.TunnellingResult(
            x, dx, k**2 / 2, potential, times, history[:, :, column]
        )
        _, above = solution.incident_tail_probabilities(sigma=sigma)
        budget = solution.region_probabilities(result)
        rows.append(
            f"| {sigma:g} | {1 / (2 * sigma):.6f} | {1.5**2 / 2 + 1 / (8 * sigma**2):.6f} | {above:.3e} | {float(budget[-1, 2]):.9f} | {float(budget[-1, 1]):.3e} | {float(solution.boundary_probability(result).max()):.3e} |"
        )
    rows += [
        "",
        "Narrowing the incident spectrum changes the physical packet and its transmission, not a discretization error.",
        "The above-barrier contribution to asymptotic transmission cannot exceed its incident weight.",
        "Do not equate the total right probability with the transmission at the central energy.",
    ]
    return rows


def measurement_study() -> list[str]:
    result = solution.solve_tunnelling(num_steps=1700, store_every=100)
    regions = solution.region_probabilities(result)
    edge = torch.cummax(solution.boundary_probability(result), dim=0).values
    rows = [
        "## Measurement window",
        "",
        "Default packet/box/grid, dt=0.02, near region [-4,4].",
        "The Gaussian barrier has no sharp end; V(4)=V0 exp(-12.5), so region cuts are diagnostics.",
        "",
        "| Time | Left | Near | Right | Maximum saved edge mass so far |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, time in enumerate(result.times):
        if float(time) == 0 or float(time) >= 16:
            left, near, right = regions[index].tolist()
            rows.append(
                f"| {float(time):g} | {left:.9f} | {near:.3e} | {right:.9f} | {float(edge[index]):.3e} |"
            )
    negative, above = solution.incident_tail_probabilities()
    rows += [
        "",
        f"Default incoming P(k<0)={negative:.3e}, P(E>V0)={above:.3e}.",
        "Choose a window where near mass is small, right mass has plateaued, and boundary effects are negligible.",
        "A longer run improves separation but eventually increases periodic-boundary contamination.",
    ]
    return rows


def write_report(output: Path) -> None:
    rows = [
        "# Quantum-tunnelling numerical validation",
        "",
        f"Generated by tunnelling_convergence.py with PyTorch {torch.__version__}, CPU float64/complex128.",
        "All experiments use m=hbar=1 and V(x)=2.5 exp[-x^2/(2*0.8^2)].",
        "Default packet: x0=-20, sigma=3, k0=1.5. L denotes the half extent of [-L,L).",
        "These are accuracy experiments, not performance benchmarks.",
        "",
    ]
    for study in (
        time_step_study,
        scattering_time_study,
        grid_study,
        domain_study,
        packet_width_study,
        measurement_study,
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
