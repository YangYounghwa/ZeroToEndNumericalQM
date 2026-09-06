"""Starter template for the PyTorch harmonic-oscillator solver."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HarmonicOscillatorResult:
    grid: Tensor
    spacing: float
    potential: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


def make_grid(
    num_points: int,
    x_max: float = 8.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    """Return the float64 interior grid and its spacing."""
    # TODO: Validate inputs and create every tensor on device.
    raise NotImplementedError


def harmonic_potential(
    grid: Tensor,
    mass: float = 1.0,
    omega: float = 1.0,
) -> Tensor:
    """Evaluate the potential on the grid."""
    # TODO: Use torch operations and preserve the grid's device and dtype.
    raise NotImplementedError


def build_hamiltonian(
    num_points: int,
    x_max: float = 8.0,
    mass: float = 1.0,
    omega: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor, Tensor]:
    """Build the Hamiltonian on the selected device."""
    # TODO: Build T and add V only to its diagonal.
    # TODO: Keep every tensor float64 and on the same device.
    raise NotImplementedError


def normalize_wavefunctions(wavefunctions: Tensor, spacing: float) -> Tensor:
    """Normalize wavefunction columns with a discrete spatial integral."""
    # TODO: Use torch operations only.
    raise NotImplementedError


def solve_harmonic_oscillator(
    num_points: int = 200,
    num_states: int = 4,
    x_max: float = 8.0,
    mass: float = 1.0,
    omega: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> HarmonicOscillatorResult:
    """Calculate the lowest-energy eigenstates."""
    # TODO: Solve with torch.linalg.eigh and normalize the requested states.
    # TODO: Do not move the result off device.
    raise NotImplementedError


def analytical_energies(
    num_states: int,
    omega: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return exact energies as float64 tensors on device."""
    # TODO: Start the quantum numbers at zero.
    raise NotImplementedError


def expectation_x_power(result: HarmonicOscillatorResult, power: int) -> Tensor:
    """Return <x^power> for all stored states."""
    # TODO: Integrate with torch.sum over the spatial dimension.
    raise NotImplementedError


def residual_norms(result: HarmonicOscillatorResult) -> Tensor:
    # TODO: Return sqrt(dx * sum(abs(H @ psi - psi * E)^2)) for each column.
    # This is an algebraic error, distinct from grid and domain errors.
    raise NotImplementedError


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"selected device: {device}")
    # TODO: Solve the problem and print a compact comparison table.
    raise NotImplementedError


if __name__ == "__main__":
    main()
