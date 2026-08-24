"""Starter template for the PyTorch infinite-square-well solver.

Complete the TODO sections after verifying PyTorch with ``torch_smoke_test.py``.
Then compare your implementation with
``solutions/infinite_square_well_torch_solution.py``.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class InfiniteWellResult:
    grid: Tensor
    spacing: float
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


def make_grid(
    num_points: int,
    length: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    """Return the float64 interior grid and its spacing."""
    # TODO: Validate num_points and length.
    # TODO: Create the grid with dtype=torch.float64 on device.
    raise NotImplementedError


def build_hamiltonian(
    num_points: int,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor]:
    """Build the finite-difference Hamiltonian on the selected device."""
    # TODO: Create all tensors with float64 on the same device.
    # TODO: Construct the tridiagonal second-derivative matrix.
    # TODO: Construct the Hamiltonian.
    raise NotImplementedError


def normalize_wavefunctions(wavefunctions: Tensor, spacing: float) -> Tensor:
    """Normalize wavefunction columns with a discrete spatial integral."""
    # TODO: Use torch operations only; do not convert to NumPy here.
    raise NotImplementedError


def solve_infinite_well(
    num_points: int = 200,
    num_states: int = 4,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> InfiniteWellResult:
    """Calculate the lowest-energy eigenstates."""
    # TODO: Build H and solve it with torch.linalg.eigh.
    # TODO: Normalize the requested eigenvectors.
    # TODO: Return an InfiniteWellResult without moving data off device.
    raise NotImplementedError


def analytical_energies(
    num_states: int,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return analytical energies as float64 tensors on device."""
    # TODO: Implement the analytical energy equation with torch operations.
    raise NotImplementedError


def main() -> None:
    # Start on CPU. Use CUDA only after the smoke test confirms it is available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TODO: Solve the problem and print numerical and analytical energies.
    print(f"selected device: {device}")
    raise NotImplementedError


if __name__ == "__main__":
    main()
