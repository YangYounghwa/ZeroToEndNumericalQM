"""Write finite-size and analytical checks; no artificial dt refinement."""

import argparse
from math import pi, sin
from pathlib import Path

import numpy as np
import torch
import two_particles_numpy_solution as reference
import two_particles_torch_solution as solution

STATISTICS: tuple[solution.Statistics, ...] = ("distinguishable", "boson", "fermion")


def spectrum_study() -> list[str]:
    rows = [
        "## Finite-chain spectra",
        "",
        "Open chain, J=1, zero site energies. Reference: allowed sums of exact sine-mode energies.",
        "",
        "| Sites | Statistics | Dimension | Ground energy | Max spectral error | NumPy matrix difference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for num_sites in (2, 3, 4, 6, 8):
        h = solution.single_particle_hamiltonian(num_sites)
        one = -2 * torch.cos(
            pi * torch.arange(1, num_sites + 1, dtype=torch.float64) / (num_sites + 1)
        )
        for statistics in STATISTICS:
            h2 = solution.two_particle_hamiltonian(h, statistics)
            energies = torch.linalg.eigvalsh(h2)
            expected = (
                torch.stack(
                    [
                        one[i] + one[j]
                        for i, j in solution.basis_pairs(num_sites, statistics)
                    ]
                )
                .sort()
                .values
            )
            error = float((energies - expected).abs().max())
            other = reference.two_particle_hamiltonian(h.numpy(), statistics)
            difference = float(np.max(np.abs(h2.numpy() - other)))
            rows.append(
                f"| {num_sites} | {statistics} | {len(energies)} | {float(energies[0]):.12f} | {error:.3e} | {difference:.3e} |"
            )
    return rows


def size_study() -> list[str]:
    rows = [
        "## Physical finite-size scaling",
        "",
        "J=1 and lattice spacing are fixed. Increasing L lengthens the chain.",
        "The full two-particle basis is used at every L. This is not a mesh-refinement error.",
        "Both ground energies approach -4 as L grows with particle number fixed at two.",
        "",
        "| L | Boson E0 | Fermion E0 | (L+1)^2*(boson E0+4) | (L+1)^2*(fermion E0+4) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for num_sites in (2, 4, 8, 12):
        h = solution.single_particle_hamiltonian(num_sites)
        boson = float(
            torch.linalg.eigvalsh(solution.two_particle_hamiltonian(h, "boson"))[0]
        )
        fermion = float(
            torch.linalg.eigvalsh(solution.two_particle_hamiltonian(h, "fermion"))[0]
        )
        rows.append(
            f"| {num_sites} | {boson:.12f} | {fermion:.12f} | {(num_sites + 1) ** 2 * (boson + 4):.8f} | {(num_sites + 1) ** 2 * (fermion + 4):.8f} |"
        )
    rows += [
        "",
        f"Predicted scaled limits: bosons 2*pi^2={2 * pi * pi:.8f}; fermions 5*pi^2={5 * pi * pi:.8f}.",
        "The finite-size shift is distinct from the small spectral errors in the preceding table.",
    ]
    return rows


def two_site_study() -> list[str]:
    rows = [
        "## Two-site pair correlations",
        "",
        "Initial state: one particle on each site. J=hbar=1; onsite energies are zero.",
        "Analytical same-site probability: sin(2t)^2/2 for labeled particles, sin(2t)^2 for bosons, zero for fermions.",
        "",
        "| t/pi | Statistics | Same-site probability | Analytical | Mean n0 | Mean n1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    times = pi * torch.tensor([0, 0.125, 0.25, 0.5], dtype=torch.float64)
    h = solution.single_particle_hamiltonian(2)
    max_error = 0.0
    for statistics in STATISTICS:
        initial = solution.localized_state(2, statistics, (0, 1))
        history = solution.evolve(
            solution.two_particle_hamiltonian(h, statistics), initial, times
        )
        same = solution.coincidence_probability(history, 2, statistics)
        occupations = solution.site_occupations(history, 2, statistics)
        factor = {"distinguishable": 0.5, "boson": 1.0, "fermion": 0.0}[statistics]
        for time, probability, counts in zip(times, same, occupations, strict=True):
            exact = factor * sin(2 * float(time)) ** 2
            max_error = max(max_error, abs(float(probability) - exact))
            rows.append(
                f"| {float(time) / pi:g} | {statistics} | {float(probability):.12f} | {exact:.12f} | {float(counts[0]):.12f} | {float(counts[1]):.12f} |"
            )
    rows += [
        "",
        f"Maximum probability error: {max_error:.3e}.",
        "Mean site counts are identical, but pair correlations differ. No interaction was added.",
    ]
    return rows


def independent_study() -> list[str]:
    rows = [
        "## Complex hopping and batched evolution",
        "",
        "Four sites. Seed 74 gives a complex Hermitian one-particle matrix, including long-range hopping.",
        "Six normalized complex initial states use batch shape (2,3). Times are [0,0.17,0.91]; hbar=1.3.",
        "PyTorch uses projection and matrix exponentials. NumPy uses occupation moves and spectral evolution.",
        "",
        "| Statistics | Matrix difference | State difference | Norm error | Energy drift | Particle-count error | Reverse error | Exchange error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    generator = torch.Generator().manual_seed(74)
    raw = torch.randn(4, 4, dtype=torch.complex128, generator=generator)
    h = (raw + raw.mH) / 2
    times = torch.tensor([0, 0.17, 0.91], dtype=torch.float64)
    for statistics in STATISTICS:
        h2 = solution.two_particle_hamiltonian(h, statistics)
        other = reference.two_particle_hamiltonian(h.numpy(), statistics)
        states = torch.randn(
            2, 3, h2.shape[0], dtype=torch.complex128, generator=generator
        )
        states /= torch.linalg.vector_norm(states, dim=-1, keepdim=True)
        history = solution.evolve(h2, states, times, hbar=1.3)
        comparison = reference.evolve(other, states.numpy(), times.numpy(), hbar=1.3)
        energy = torch.einsum("...i,ij,...j->...", history.conj(), h2, history).real
        counts = solution.site_occupations(history, 4, statistics).sum(-1)
        reverse = solution.evolve(h2, history[-1], -times[-1:], hbar=1.3)[0]
        values = [
            float(np.max(np.abs(h2.numpy() - other))),
            float(np.max(np.abs(history.numpy() - comparison))),
            float((history.abs().square().sum(-1) - 1).abs().max()),
            float((energy - energy[0]).abs().max()),
            float((counts - 2).abs().max()),
            float((reverse - states).abs().max()),
        ]
        exchange = "not required"
        if statistics != "distinguishable":
            amplitudes = solution.pair_amplitudes(history, 4, statistics)
            sign = 1 if statistics == "boson" else -1
            exchange = f"{float((amplitudes.transpose(-1, -2) - sign * amplitudes).abs().max()):.3e}"
        rows.append(
            f"| {statistics} | "
            + " | ".join(f"{value:.3e}" for value in values)
            + f" | {exchange} |"
        )
    rows += [
        "",
        "Each requested time is evaluated directly. These are floating-point checks, not dt convergence.",
        "Tests also cover canonical basis order, double occupancy, uniform energy shifts, invalid inputs, and batch/single agreement.",
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "VALIDATION.md",
    )
    arguments = parser.parse_args()
    rows = [
        "# Two-particle lattice validation",
        "",
        f"Generated with PyTorch {torch.__version__} and NumPy {np.__version__} on CPU in float64/complex128.",
        "No interactions, spatial quadrature, or basis cutoff are used.",
        "",
    ]
    for section in (
        spectrum_study(),
        size_study(),
        two_site_study(),
        independent_study(),
    ):
        rows.extend(section)
        rows.append("")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
