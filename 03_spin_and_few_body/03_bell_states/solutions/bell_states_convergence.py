"""Write analytical parameter and floating-point studies to Markdown."""

import argparse
from math import log2, pi
from pathlib import Path

import bell_states_numpy_solution as reference
import bell_states_torch_solution as solution
import torch


def binary_entropy(probability: float) -> float:
    """Independent scalar reference, including both endpoints."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    return -sum(p * log2(p) for p in (probability, 1 - probability) if p > 0)


def state_study() -> list[str]:
    basis = torch.eye(4, dtype=torch.complex128)
    mixture = solution.mixed_density(
        basis[[0, 3]], torch.tensor([0.5, 0.5], dtype=torch.float64)
    )
    states = [("Product 00", solution.pure_density(basis[0]))]
    states += list(
        zip(
            ("Phi+", "Phi-", "Psi+", "Psi-"),
            solution.pure_density(solution.bell_states()),
            strict=True,
        )
    )
    states.append(("Classical 00/11 mixture", mixture))
    rows = [
        "## Joint and local states",
        "",
        "| State | Joint purity | Joint S (bits) | Local purity A | S(A) | S(B) | xx | yy | zz |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, rho in states:
        a, b = solution.partial_trace(rho), solution.partial_trace(rho, keep=1)
        corr = solution.pauli_correlations(rho)
        values = [
            solution.purity(rho),
            solution.entropy(rho),
            solution.purity(a),
            solution.entropy(a),
            solution.entropy(b),
            corr[0, 0],
            corr[1, 1],
            corr[2, 2],
        ]
        rows.append(
            f"| {name} | " + " | ".join(f"{float(v):.6f}" for v in values) + " |"
        )
    rows += [
        "",
        "Bell states and the classical mixture have the same local states I/2.",
        "The mixture is explicitly separable; its local entropy of one bit is not an entanglement measure.",
        "Phi+ and the mixture share computational-basis probabilities and zz=1, but differ in xx and yy.",
    ]
    return rows


def schmidt_study() -> list[str]:
    angles = pi * torch.tensor(
        [0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5], dtype=torch.float64
    )
    states = solution.schmidt_states(angles, phase=0.73)
    values = solution.entanglement_entropy(states)
    rows = [
        "## Schmidt-angle parameter study",
        "",
        "State: cos(theta)|00> + exp(0.73i) sin(theta)|11>.",
        "Eigenvalues of either reduced state are cos(theta)^2 and sin(theta)^2.",
        "Changing theta changes the physical state; it is not numerical refinement.",
        "",
        "| theta/pi | S(A) | Analytical binary entropy | Absolute error |",
        "| --- | --- | --- | --- |",
    ]
    for theta, value in zip(angles, values, strict=True):
        expected = binary_entropy(float(torch.sin(theta).square()))
        rows.append(
            f"| {float(theta) / pi:g} | {float(value):.12f} | {expected:.12f} | {abs(float(value) - expected):.3e} |"
        )
    return rows


def exchange_study() -> list[str]:
    times = pi * torch.tensor([0, 0.25, 0.5, 0.75, 1, 1.5, 2], dtype=torch.float64)
    states = solution.exchange_states(times)
    values = solution.entanglement_entropy(states)
    matrix = torch.tensor(
        [[0.25, 0, 0, 0], [0, -0.25, 0.5, 0], [0, 0.5, -0.25, 0], [0, 0, 0, 0.25]],
        dtype=torch.complex128,
    )
    numerical = torch.linalg.matrix_exp(
        -1j * times[:, None, None] * matrix
    ) @ torch.tensor([0, 1, 0, 0], dtype=torch.complex128)
    error = float(
        (solution.pure_density(numerical) - solution.pure_density(states)).abs().max()
    )
    rows = [
        "## Connection to exchange dynamics",
        "",
        "J=hbar=1, zero fields, initial |01>. Exact entropy is h2(sin(t/2)^2).",
        f"Maximum density-matrix entry error against a PyTorch matrix exponential: {error:.3e}.",
        "Global phases cancel in density matrices. These are exact-time samples, not integration steps.",
        "",
        "| t/pi | P(10) | S(A) | Analytical entropy |",
        "| --- | --- | --- | --- |",
    ]
    for time, value in zip(times, values, strict=True):
        probability = float(torch.sin(time / 2).square())
        rows.append(
            f"| {float(time) / pi:g} | {probability:.6f} | {float(value):.6f} | {binary_entropy(probability):.6f} |"
        )
    rows += [
        "",
        "The state is maximally entangled halfway to a complete swap (t=pi/2).",
        "At the complete swap t=pi, it is a product state again.",
    ]
    return rows


def precision_study() -> list[str]:
    rows = [
        "## Small eigenvalues and floating-point error",
        "",
        "Density matrix diag(1-p,p). Do not discard small positive eigenvalues using the positivity tolerance.",
        "Zero eigenvalues contribute exactly zero; only negative roundoff within the validation tolerance is clipped.",
        "",
        "| p | PyTorch entropy | Scalar reference | Absolute error |",
        "| --- | --- | --- | --- |",
    ]
    for small in (0, 1e-14, 1e-10, 1e-6, 0.01, 0.5):
        rho = torch.diag(torch.tensor([1 - small, small], dtype=torch.complex128))
        value, expected = float(solution.entropy(rho)), binary_entropy(small)
        rows.append(
            f"| {small:g} | {value:.12e} | {expected:.12e} | {abs(value - expected):.3e} |"
        )
    rows += [
        "",
        "A rotated rank-one projector can acquire eigenvalues of either sign at roundoff scale.",
        "The positivity tolerance (1e-12) permits small numerical defects, but does not repair a physical state.",
        "Entropy near zero should be interpreted with an absolute error tolerance.",
        "No spatial grid, truncated basis, or time integrator is used in this chapter, so there is no dt/dx convergence rate to report.",
    ]
    return rows


def library_study() -> list[str]:
    generator = torch.Generator().manual_seed(2026)
    states = torch.randn(2, 3, 5, 4, generator=generator, dtype=torch.complex128)
    states /= torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    weights = torch.rand(2, 3, 5, generator=generator, dtype=torch.float64)
    weights /= weights.sum(-1, keepdim=True)
    rho = solution.mixed_density(states, weights)
    other = reference.mixed_density(states.numpy(), weights.numpy())
    errors = {"Density matrix": float((rho - torch.from_numpy(other)).abs().max())}
    for keep in (0, 1):
        local = solution.partial_trace(rho, keep=keep)
        expected = reference.partial_trace(other, keep=keep)
        errors[f"Reduced matrix {keep}"] = float(
            (local - torch.from_numpy(expected)).abs().max()
        )
        errors[f"Reduced entropy {keep}"] = float(
            (solution.entropy(local) - torch.from_numpy(reference.entropy(expected)))
            .abs()
            .max()
        )
    errors["Joint purity"] = float(
        (solution.purity(rho) - torch.from_numpy(reference.purity(other))).abs().max()
    )
    errors["Pauli correlations"] = float(
        (
            solution.pauli_correlations(rho)
            - torch.from_numpy(reference.pauli_correlations(other))
        )
        .abs()
        .max()
    )
    trace_error = float((rho.diagonal(dim1=-2, dim2=-1).sum(-1) - 1).abs().max())
    rows = [
        "## Batched NumPy comparison",
        "",
        "Seed 2026; batch shape (2,3), five random normalized pure states per mixture.",
        "NumPy partial traces use index loops, independently checking the PyTorch contractions.",
        f"Maximum trace error: {trace_error:.3e}; minimum joint eigenvalue: {float(torch.linalg.eigvalsh(rho).min()):.3e}.",
        "",
        "| Quantity | Maximum absolute library difference |",
        "| --- | --- |",
    ]
    rows += [f"| {name} | {value:.3e} |" for name, value in errors.items()]
    return rows


def write_report(path: Path) -> None:
    rows = [
        "# Bell states validation",
        "",
        f"Generated on CPU with PyTorch {torch.__version__} using float64/complex128.",
        "Regenerate from the repository root:",
        "",
        "```powershell",
        "uv run python 03_spin_and_few_body/03_bell_states/solutions/bell_states_convergence.py",
        "```",
        "",
    ]
    for study in (
        state_study,
        schmidt_study,
        exchange_study,
        precision_study,
        library_study,
    ):
        rows += [*study(), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


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
