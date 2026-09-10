"""Write two-spin analytical comparisons and time-error studies to Markdown."""

import argparse
from math import pi, sqrt
from pathlib import Path

import coupled_spins_numpy_solution as reference
import coupled_spins_torch_solution as solution
import torch


def spectrum_study() -> list[str]:
    rows = [
        "## Zero-field spectrum",
        "",
        "The triplet has energy J/4; the singlet has energy -3J/4. hbar=1.",
        "Residuals use the explicitly derived singlet/triplet states, not numerically selected eigenvectors.",
        "",
        "| J | Triplet E | Singlet E | Maximum eigenpair residual | Sorted numerical energies |",
        "| --- | --- | --- | --- | --- |",
    ]
    basis = solution.singlet_triplet_basis()
    for exchange in (-1.0, 0.0, 1.0, 2.0):
        matrix = solution.hamiltonians(
            torch.tensor([exchange], dtype=torch.float64),
            torch.zeros((1, 2, 3), dtype=torch.float64),
        )[0]
        energies = torch.tensor(
            [exchange / 4] * 3 + [-3 * exchange / 4], dtype=torch.float64
        )
        residual = float((matrix @ basis - basis * energies).abs().max())
        spectrum = ", ".join(f"{float(e):.6f}" for e in torch.linalg.eigvalsh(matrix))
        rows.append(
            f"| {exchange:g} | {exchange / 4:.6f} | {-3 * exchange / 4:.6f} | {residual:.3e} | {spectrum} |"
        )
    return rows


def exchange_dynamics_study() -> list[str]:
    times = pi * torch.tensor([0, 0.25, 0.5, 0.75, 1, 1.5, 2], dtype=torch.float64)
    initial = torch.tensor([[0, 1, 0, 0]], dtype=torch.complex128)
    matrix = solution.hamiltonians(
        torch.ones(1, dtype=torch.float64), torch.zeros((1, 2, 3), dtype=torch.float64)
    )
    result = solution.evolve_constant(initial, matrix, times)
    expected = solution.exchange_reference(times)
    error = float(solution.phase_aligned_errors(result.states[:, 0], expected).max())
    local = solution.local_bloch_vectors(result.states)[:, 0]
    connected = solution.correlations(result.states, connected=True)[:, 0]
    probabilities = solution.joint_probabilities(result.states)[:, 0]
    determinant = solution.product_determinant(result.states)[:, 0]
    rows = [
        "## Exchange dynamics and correlations",
        "",
        "J=hbar=1, no fields, initial |+->. Exact swapped probability is sin(t/2)^2.",
        f"Maximum phase-aligned state error to the analytical solution: {error:.3e}.",
        "Local z columns are Pauli expectations; multiply by hbar/2 for spin angular momentum.",
        "",
        "| Time/pi | P(+-) | P(-+) | Local z1 | Local z2 | Connected zz | Product determinant |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, time in enumerate(times):
        rows.append(
            f"| {float(time) / pi:g} | {float(probabilities[index, 1]):.6f} | {float(probabilities[index, 2]):.6f} | {float(local[index, 0, 2]):.6f} | {float(local[index, 1, 2]):.6f} | {float(connected[index, 2, 2]):.6f} | {float(determinant[index]):.6f} |"
        )
    rows += [
        "",
        "At t=pi/2, local Bloch vectors vanish but joint correlations do not.",
        "The coefficient determinant is 1/2, so this pure state cannot be written as a product of two spinors.",
        "At t=pi a complete swap has occurred and the state is a product again.",
    ]
    return rows


def detuning_study() -> list[str]:
    rows = [
        "## Unequal longitudinal fields",
        "",
        "J=gamma=hbar=1, B1z=delta/2 and B2z=-delta/2; initial |+->.",
        "For each delta, sample the first transfer peak t=pi/sqrt(J^2+delta^2) exactly.",
        "delta is an energy difference, not a numerical discretization parameter.",
        "",
        "| delta | First peak time | Predicted maximum transfer | Numerical transfer | State error to block solution |",
        "| --- | --- | --- | --- | --- |",
    ]
    initial = torch.tensor([[0, 1, 0, 0]], dtype=torch.complex128)
    for delta in (0.0, 0.5, 1.0, 2.0):
        fields = torch.zeros((1, 2, 3), dtype=torch.float64)
        fields[0, 0, 2], fields[0, 1, 2] = delta / 2, -delta / 2
        time = pi / sqrt(1 + delta**2)
        times = torch.tensor([0, time], dtype=torch.float64)
        result = solution.evolve_constant(
            initial, solution.hamiltonians(torch.ones(1), fields), times
        )
        expected = solution.exchange_reference(times, detuning=delta)
        error = float(
            solution.phase_aligned_errors(result.states[:, 0], expected).max()
        )
        transfer = float(solution.joint_probabilities(result.states)[-1, 0, 2])
        rows.append(
            f"| {delta:g} | {time:.6f} | {1 / (1 + delta**2):.6f} | {transfer:.6f} | {error:.3e} |"
        )
    return rows


def library_study() -> list[str]:
    coupling = torch.tensor([0, -0.7, 1.3], dtype=torch.float64)
    fields = torch.tensor(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0.3, -0.4, 1], [-0.2, 0.7, 0.1]],
            [[0, 0, 1], [0, 0, -0.4]],
        ],
        dtype=torch.float64,
    )
    initial = solution.normalize_states(
        torch.tensor(
            [[0, 1, 0, 0], [1, 1j, -0.4, 0.7], [1, 0, 0, 1]], dtype=torch.complex128
        )
    )
    times = torch.linspace(-1, 9, 41, dtype=torch.float64)
    matrix = solution.hamiltonians(coupling, fields, gamma=-0.8, hbar=0.7)
    result = solution.evolve_constant(initial, matrix, times, hbar=0.7)
    numpy_matrix = reference.hamiltonians(
        coupling.numpy(), fields.numpy(), gamma=-0.8, hbar=0.7
    )
    numpy_result = reference.evolve_constant(
        initial.numpy(), numpy_matrix, times.numpy(), hbar=0.7
    )
    errors = solution.phase_aligned_errors(
        result.states, torch.from_numpy(numpy_result.states)
    )
    energies = solution.energy_expectations(result)
    correlation_error = (
        solution.correlations(result.states, connected=True)
        - torch.from_numpy(reference.correlations(numpy_result.states, connected=True))
    ).abs()
    rows = [
        "## General-field NumPy comparison",
        "",
        "Three independent parameter/state pairs; gamma=-0.8, hbar=0.7, times -1 through 9.",
        "Cases: zero Hamiltonian, unequal noncollinear fields, and unequal longitudinal fields.",
        "NumPy uses eigh; PyTorch uses native matrix_exp. Errors are maxima over time.",
        "",
        "| Case | State error | Connected-correlation difference | Norm error | Energy drift |",
        "| --- | --- | --- | --- | --- |",
    ]
    for batch in range(3):
        rows.append(
            f"| {batch + 1} | {float(errors[:, batch].max()):.3e} | {float(correlation_error[:, batch].max()):.3e} | {float((result.states[:, batch].abs().square().sum(-1) - 1).abs().max()):.3e} | {float((energies[:, batch] - energies[0, batch]).abs().max()):.3e} |"
        )
    return rows


def symmetry_study() -> list[str]:
    first, second = solution.local_operators()
    total_z = (first[2] + second[2]) / 2
    total_squared = (((first + second) / 2) @ ((first + second) / 2)).sum(0)
    cases = [
        ("Zero field", [[0, 0, 0], [0, 0, 0]]),
        ("Uniform z", [[0, 0, 1], [0, 0, 1]]),
        ("Uniform tilted", [[0.3, -0.4, 1], [0.3, -0.4, 1]]),
        ("Unequal z", [[0, 0, 1], [0, 0, -0.4]]),
        ("Unequal transverse", [[1, 0, 0], [0, 0, 0]]),
    ]
    rows = [
        "## Which quantities are conserved?",
        "",
        "J=hbar=gamma=1. Each entry is the maximum absolute matrix element of a commutator.",
        "Zero commutator implies conservation for every initial state in this static model.",
        "S_total,z is in hbar units and S_total^2 in hbar^2 units.",
        "",
        "| Fields | [H,S_total,z] | [H,S_total^2] |",
        "| --- | --- | --- |",
    ]
    for label, field in cases:
        matrix = solution.hamiltonians(
            torch.ones(1), torch.tensor([field], dtype=torch.float64)
        )[0]
        rows.append(
            f"| {label} | {float((matrix @ total_z - total_z @ matrix).abs().max()):.3e} | {float((matrix @ total_squared - total_squared @ matrix).abs().max()):.3e} |"
        )
    return rows


def time_step_study() -> list[str]:
    rows = [
        "## Crank-Nicolson time error",
        "",
        "J=hbar=1, zero fields, initial |+->, final time=10. Exact reference: exchange block solution.",
        "State and swapped-probability errors are at the final time; invariant errors cover all steps.",
        "",
        "| dt | State error | Previous/current error | Swap-probability error | Norm error | Energy drift |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    initial = torch.tensor([[0, 1, 0, 0]], dtype=torch.complex128)
    matrix = solution.hamiltonians(torch.ones(1), torch.zeros((1, 2, 3)))
    previous: float | None = None
    for dt in (0.4, 0.2, 0.1, 0.05):
        result = solution.propagate_cn(initial, matrix, dt, round(10 / dt))
        exact = solution.exchange_reference(result.times)
        error = float(solution.phase_aligned_errors(result.states[-1, 0], exact[-1]))
        ratio = "—" if previous is None else f"{previous / error:.3f}"
        probability_error = abs(
            float(result.states[-1, 0, 2].abs().square() - exact[-1, 2].abs().square())
        )
        energies = solution.energy_expectations(result)
        rows.append(
            f"| {dt:g} | {error:.3e} | {ratio} | {probability_error:.3e} | {float((result.states.abs().square().sum(-1) - 1).abs().max()):.3e} | {float((energies - energies[0]).abs().max()):.3e} |"
        )
        previous = error
    rows += [
        "",
        "Norm and energy conservation do not guarantee correct exchange timing.",
        "The four-state spin model has no spatial or basis-cutoff error; the CN time error is studied separately from physical detuning.",
    ]
    return rows


def write_report(output: Path) -> None:
    rows = [
        "# Two coupled spins: numerical validation",
        "",
        f"Generated with PyTorch {torch.__version__}, CPU float64/complex128.",
        "H=J*sigma1.sigma2/4-gamma*hbar*(B1.sigma1+B2.sigma2)/2.",
        "Basis order (++,+-,-+,--); J and detuning have energy units.",
        "",
    ]
    for study in (
        spectrum_study,
        exchange_dynamics_study,
        detuning_study,
        library_study,
        symmetry_study,
        time_step_study,
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
