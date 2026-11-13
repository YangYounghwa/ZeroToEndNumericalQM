"""Generate exact dimer checks and a large-U approximation study."""

import argparse
from math import sqrt
from pathlib import Path

import hubbard_numpy_solution as reference
import hubbard_torch_solution as solution
import numpy as np
import torch


def report() -> str:
    rows = [
        "# Hubbard dimer validation",
        "",
        f"CPU: PyTorch {torch.__version__}, NumPy {np.__version__}; float64/complex128.",
        "",
        "## Symmetric dimer and exchange approximation",
        "",
        "Hopping is 1. The full sector is retained. U changes the model, not the numerical resolution.",
        "For positive U, J is the exact singlet-triplet gap. Compare 4/U with J.",
        "",
        "| U | E0 | Energy error | Double occupancy | D error | Relative gap approximation error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for u in (-8.0, 0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        h = solution.hamiltonian(interaction=u)
        energies, vectors = torch.linalg.eigh(h)
        root = sqrt(u * u + 16)
        exact = -8 / (root + u)
        d = float(solution.double_occupancy(vectors[:, 0]))
        gap_error = f"{abs((4 / u) / (-exact) - 1):.6e}" if u > 0 else "not applicable"
        rows.append(
            f"| {u:g} | {float(energies[0]):.12f} | {abs(float(energies[0]) - exact):.3e} | {d:.12f} | {abs(d - (1 - u / root) / 2):.3e} | {gap_error} |"
        )
    rows += [
        "",
        "The leading relative gap error is proportional to 1/U^2 at fixed hopping.",
        "This approximation error is distinct from eigensolver roundoff.",
        "",
        "## Biased dimer dynamics",
        "",
        "U=3.1, bias=0.8, hopping=1. Seed 52. Six states use batch shape (2,3).",
        "Times are 0, 0.3, and 1.7. No integration time step is used.",
        "",
    ]
    h = solution.hamiltonian(interaction=3.1, bias=0.8)
    other = reference.hamiltonian(interaction=3.1, bias=0.8)
    states = torch.randn(
        2, 3, 4, dtype=torch.complex128, generator=torch.Generator().manual_seed(52)
    )
    states /= torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    times = torch.tensor([0.0, 0.3, 1.7], dtype=torch.float64)
    history = solution.evolve(h, states, times)
    comparison = reference.evolve(other, states.numpy(), times.numpy())
    energy = torch.einsum("...i,ij,...j->...", history.conj(), h, history)
    energies, vectors = torch.linalg.eigh(h)
    errors = {
        "Independent matrix difference": float(np.max(np.abs(h.numpy() - other))),
        "Independent state difference": float(
            np.max(np.abs(history.numpy() - comparison))
        ),
        "Eigenpair residual": float((h @ vectors - vectors * energies).abs().max()),
        "Norm error": float((history.abs().square().sum(-1) - 1).abs().max()),
        "Energy drift": float((energy - energy[0]).abs().max()),
        "Particle-count error": float(
            (solution.site_occupations(history).sum(-1) - 2).abs().max()
        ),
        "Reverse-state error": float(
            (solution.evolve(h, history[-1], -times[-1:])[0] - states).abs().max()
        ),
    }
    for label, error in errors.items():
        rows.append(f"- {label}: {error:.3e}.")
    rows += [
        "",
        "Tests also check the full fermion algebra, atomic and noninteracting limits,",
        "exact singlet doublon oscillations, a stationary triplet, and invalid inputs.",
        "CUDA is tested only when available. No bulk-limit claim follows from a dimer.",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "VALIDATION.md",
    )
    args = parser.parse_args()
    args.output.write_text(report(), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
