"""Generate finite-size results and separate sparse solver accuracy checks."""

import argparse
from math import pi, sin
from pathlib import Path

import ising_numpy_solution as reference
import ising_torch_solution as solution
import numpy as np
import torch


def report() -> str:
    rows = [
        "# Transverse-field Ising validation",
        "",
        f"CPU: PyTorch {torch.__version__}, NumPy {np.__version__}. Real matrices use float64.",
        "",
        "## Critical open chains",
        "",
        "J=g=1. The full finite spin space is retained. Compare with exact open-chain formulas.",
        "L changes the physical model. Residuals and energy errors measure numerical accuracy.",
        "",
        "| L | D | E0/L | L gap | Ground energy error | Gap error | Max sparse residual | Dense comparison error |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for sites in (2, 4, 6, 8, 10, 12):
        energies, _, residuals = reference.low_energy(sites)
        angle = pi / (4 * sites + 2)
        gap = float(energies[1] - energies[0])
        dense = "not run"
        if sites <= 8:
            values = torch.linalg.eigvalsh(solution.hamiltonian(sites))[:2].numpy()
            dense = f"{float(np.max(np.abs(values - energies))):.3e}"
        rows.append(
            f"| {sites} | {2**sites} | {energies[0] / sites:.10f} | {sites * gap:.10f} | {abs(energies[0] - (1 - 1 / sin(angle))):.3e} | {abs(gap - 4 * sin(angle)):.3e} | {float(np.max(residuals)):.3e} | {dense} |"
        )
    rows += [
        "",
        f"Bulk targets: E0/L -> {-4 / pi:.10f}; L gap -> {pi:.10f}.",
        "These limits are not expected to equal finite-chain results.",
        "",
        "## Sparse tolerance study",
        "",
        "L=10 and J=g=1 are fixed. Report actual errors; roundoff need not improve monotonically.",
        "",
        "| Requested tolerance | E0 error | Max residual |",
        "| --- | --- | --- |",
    ]
    exact = 1 - 1 / sin(pi / 42)
    for tolerance in (1e-4, 1e-8, 1e-12):
        energies, _, residuals = reference.low_energy(10, tolerance=tolerance)
        rows.append(
            f"| {tolerance:.0e} | {abs(energies[0] - exact):.3e} | {float(np.max(residuals)):.3e} |"
        )
    rows += [
        "",
        "## Fixed-size field study",
        "",
        "L=6, J=1. Mx and Mz are mean Pauli magnetizations. C is <Z0 Z5>.",
        "Finite parity eigenstates have zero Mz, even when C is large.",
        "",
        "| g | E0 | Gap | Mx | Mz | C | Ground parity |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for field in (0.25, 0.5, 1.0, 2.0):
        energies, vectors = torch.linalg.eigh(solution.hamiltonian(6, field=field))
        ground = vectors[:, 0]
        mx = float(solution.magnetization(ground, 6, "x"))
        mz = float(solution.magnetization(ground, 6, "z"))
        c = float(solution.zz_correlation(ground, 6, 0, 5))
        parity = float(solution.parity_expectation(ground, 6))
        rows.append(
            f"| {field:g} | {float(energies[0]):.10f} | {float(energies[1] - energies[0]):.6e} | {mx:.10f} | {mz:.3e} | {c:.10f} | {parity:.10f} |"
        )
    rows += [
        "",
        "## Independent matrix and symmetry checks",
        "",
        "L=5, J=0.8, g=0.6.",
        "",
    ]
    h = solution.hamiltonian(5, coupling=0.8, field=0.6)
    other = reference.hamiltonian(5, coupling=0.8, field=0.6)
    blocks = []
    for parity in (-1, 1):
        q = solution.parity_embedding(5, parity)
        block = q.T @ h @ q
        blocks.append(torch.linalg.eigvalsh(block))
        rows.append(
            f"- Parity {parity:+d} invariant-subspace residual: {float((h @ q - q @ block).abs().max()):.3e}."
        )
    error = float(
        (torch.cat(blocks).sort().values - torch.linalg.eigvalsh(h)).abs().max()
    )
    rows += [
        f"- Union-of-sector spectral error: {error:.3e}.",
        f"- Independent matrix difference: {float(np.max(np.abs(h.numpy() - other))):.3e}.",
        "",
        "Tests also cover exact limits, complex batches, local bit order, and invalid inputs.",
        "CUDA checks run only when available. Sparse residuals do not certify missed degeneracy multiplicities.",
        "No phase transition or critical exponent is inferred from these small chains.",
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
