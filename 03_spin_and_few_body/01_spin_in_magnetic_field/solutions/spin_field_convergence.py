"""Write spin-precession references and CN time-error experiments to Markdown."""

import argparse
from math import atan, pi
from pathlib import Path

import spin_field_numpy_solution as reference
import spin_field_torch_solution as solution
import torch


def constant_field_study() -> list[str]:
    rows = [
        "## Constant-field references",
        "",
        "Initial state: theta=pi/3, phi=pi/5. Times 0 through 10, 101 samples, hbar=0.7.",
        "PyTorch matrix_exp is compared with the Pauli closed form, NumPy eigh, and Rodrigues rotation.",
        "State errors remove global phase; observables use independently computed references.",
        "",
        "| Field B | gamma | Closed-form state error | NumPy state error | Bloch error | Norm error | Energy drift |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    cases = [
        ((0.0, 0.0, 0.0), 1.0),
        ((0.0, 0.0, 1.0), 1.0),
        ((1.0, 0.0, 0.0), 1.0),
        ((0.3, -0.4, 1.0), 1.0),
        ((0.3, -0.4, 1.0), -1.7),
    ]
    times = torch.linspace(0, 10, 101, dtype=torch.float64)
    initial = solution.spinor(pi / 3, pi / 5)[None, :]
    for field, gamma in cases:
        fields = torch.tensor([field], dtype=torch.float64)
        result = solution.evolve_constant(initial, fields, times, gamma, 0.7)
        closed = (
            solution.closed_form_propagators(fields, times, gamma, 0.7)
            @ initial[None, :, :, None]
        ).squeeze(-1)
        numpy_result = reference.evolve_constant(
            initial.numpy(), fields.numpy(), times.numpy(), gamma, 0.7
        )
        bloch = solution.rodrigues_bloch(
            solution.bloch_vectors(initial), fields, times, gamma
        )
        energies = solution.energy_expectations(result)
        rows.append(
            f"| {field} | {gamma:g} | {float(solution.phase_aligned_errors(result.states, closed).max()):.3e} | {float(solution.phase_aligned_errors(result.states, torch.from_numpy(numpy_result.states)).max()):.3e} | {float((solution.bloch_vectors(result.states) - bloch).abs().max()):.3e} | {float((result.states.abs().square().sum(-1) - 1).abs().max()):.3e} | {float((energies - energies[0]).abs().max()):.3e} |"
        )
    return rows


def signed_precession_study() -> list[str]:
    rows = [
        "## Signed precession and spinor phase",
        "",
        "B=(0,0,1), gamma=hbar=1, initial +x. Therefore Omega=(0,0,-1).",
        "Expected Bloch vector: (cos(t),-sin(t),0). At 2pi the spinor changes sign;",
        "at 4pi it returns. Probabilities already return after 2pi.",
        "",
        "| Time/pi | Bloch x | Bloch y | P(+x) | P(+z) | Real overlap with initial spinor |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    fields = torch.tensor([[0, 0, 1]], dtype=torch.float64)
    times = pi * torch.tensor([0, 0.5, 1, 1.5, 2, 4], dtype=torch.float64)
    initial = solution.spinor()[None, :]
    result = solution.evolve_constant(initial, fields, times)
    bloch = solution.bloch_vectors(result.states)[:, 0]
    px = solution.measurement_probabilities(
        result.states[:, 0], torch.tensor([1, 0, 0], dtype=torch.float64)
    )
    pz = solution.measurement_probabilities(result.states[:, 0], fields[0])
    overlaps = (initial.conj() * result.states[:, 0]).sum(-1)
    for index, time in enumerate(times):
        rows.append(
            f"| {float(time) / pi:g} | {float(bloch[index, 0]):.6f} | {float(bloch[index, 1]):.6f} | {float(px[index, 0]):.6f} | {float(pz[index, 0]):.6f} | {float(overlaps[index].real):.6f} |"
        )
    return rows


def time_step_study() -> list[str]:
    rows = [
        "## Crank-Nicolson time-step convergence",
        "",
        "B=(0.3,-0.4,1), gamma=hbar=1, initial +x, final time=10.",
        "Compare with the exact constant-field Pauli propagator. Save every step.",
        "State error is at the final time; Bloch, norm, and energy errors are maxima over time.",
        "",
        "| dt | State error | Previous/current error | Bloch error | Norm error | Energy drift |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    initial = solution.spinor()[None, :]
    fields = torch.tensor([[0.3, -0.4, 1]], dtype=torch.float64)
    previous: float | None = None
    for dt in (0.4, 0.2, 0.1, 0.05, 0.025):
        result = solution.propagate_cn(initial, fields, dt, round(10 / dt))
        closed = (
            solution.closed_form_propagators(fields, result.times)
            @ initial[None, :, :, None]
        ).squeeze(-1)
        error = float(solution.phase_aligned_errors(result.states[-1], closed[-1])[0])
        ratio = "—" if previous is None else f"{previous / error:.3f}"
        energies = solution.energy_expectations(result)
        rows.append(
            f"| {dt:g} | {error:.3e} | {ratio} | {float((solution.bloch_vectors(result.states) - solution.bloch_vectors(closed)).abs().max()):.3e} | {float((result.states.abs().square().sum(-1) - 1).abs().max()):.3e} | {float((energies - energies[0]).abs().max()):.3e} |"
        )
        previous = error
    rows += [
        "",
        "CN conserves norm and static-H energy, yet rotates at the wrong angular speed at finite dt.",
        "There is no spatial grid or basis truncation here: the two spin states form the complete Hilbert space.",
    ]
    return rows


def long_time_study() -> list[str]:
    rows = [
        "## Accumulated phase error",
        "",
        "B=(0,0,1), initial +x, gamma=hbar=1, fixed CN dt=0.2.",
        "The effective angular speed is omega_CN=4*atan(omega*dt/4)/dt.",
        "Unwrapped angle lag is (omega-omega_CN)*T; state error is phase-aligned and bounded.",
        "",
        "| Final time | Angle lag (rad) | State error to exact | Norm error |",
        "| --- | --- | --- | --- |",
    ]
    dt = 0.2
    speed = 4 * atan(dt / 4) / dt
    fields = torch.tensor([[0, 0, 1]], dtype=torch.float64)
    initial = solution.spinor()[None, :]
    for duration in (10, 50, 100, 200):
        steps = round(duration / dt)
        result = solution.propagate_cn(initial, fields, dt, steps, steps)
        exact = solution.evolve_constant(initial, fields, result.times)
        error = float(
            solution.phase_aligned_errors(result.states[-1], exact.states[-1])[0]
        )
        rows.append(
            f"| {duration} | {(1 - speed) * duration:.6f} | {error:.3e} | {float((result.states.abs().square().sum(-1) - 1).abs().max()):.3e} |"
        )
    rows += [
        "",
        "Longer evolution can require a smaller dt even when the norm remains excellent.",
    ]
    return rows


def measurement_study() -> list[str]:
    state = solution.spinor(pi / 3, pi / 5)
    direction = solution.bloch_vectors(state)
    axes = [
        torch.tensor([1, 0, 0], dtype=torch.float64),
        torch.tensor([0, 1, 0], dtype=torch.float64),
        torch.tensor([0, 0, 1], dtype=torch.float64),
        direction,
    ]
    rows = [
        "## Measurement directions",
        "",
        "State theta=pi/3, phi=pi/5. Outcomes are +hbar/2 and -hbar/2 along the chosen axis.",
        "",
        "| Axis | P(+) | P(-) | Sum |",
        "| --- | --- | --- | --- |",
    ]
    for label, axis in zip(
        ("x", "y", "z", "initial Bloch direction"), axes, strict=True
    ):
        probabilities = solution.measurement_probabilities(state, axis)
        rows.append(
            f"| {label} | {float(probabilities[0]):.9f} | {float(probabilities[1]):.9f} | {float(probabilities.sum()):.9f} |"
        )
    rows += [
        "",
        "Probabilities determine measurement statistics; this chapter computes them without drawing random measurement outcomes.",
    ]
    return rows


def write_report(output: Path) -> None:
    rows = [
        "# Spin-1/2 numerical validation",
        "",
        f"Generated by spin_field_convergence.py with PyTorch {torch.__version__}, CPU float64/complex128.",
        "Convention: H=-gamma*hbar*B.sigma/2, Omega=-gamma*B; gamma is signed.",
        "Numerical matrix exponentials have roundoff but no time-integration step error for this static Hamiltonian.",
        "",
    ]
    for study in (
        constant_field_study,
        signed_precession_study,
        time_step_study,
        long_time_study,
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
