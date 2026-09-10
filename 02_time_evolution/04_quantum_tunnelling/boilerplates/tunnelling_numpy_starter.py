"""Starter: FFT tunnelling. Complete a copy in workbench/."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class TunnellingResult:
    grid: FloatArray
    spacing: float
    kinetic: FloatArray
    potential: FloatArray
    times: FloatArray
    wavefunctions: ComplexArray


def make_grid(
    num_points: int = 1024, half_extent: float = 64.0
) -> tuple[FloatArray, float, FloatArray]:
    """TODO: Exclude the duplicate right endpoint; return unshifted angular wave numbers."""
    raise NotImplementedError


def gaussian_barrier(
    grid: FloatArray, height: float = 2.5, width: float = 0.8
) -> FloatArray:
    """TODO: Sample the Gaussian potential with a positive width and nonnegative height."""
    raise NotImplementedError


def gaussian_packet(
    grid: FloatArray,
    center: float = -20.0,
    sigma: float = 3.0,
    wave_number: float = 1.5,
) -> ComplexArray:
    """TODO: Combine the Gaussian envelope with a complex traveling phase."""
    raise NotImplementedError


def normalize_columns(states: ComplexArray, spacing: float) -> ComplexArray:
    """TODO: Use dx-weighted norms for (space,batch) columns, rejecting zero/nonfinite states."""
    raise NotImplementedError


def propagate_batch(
    initial_states: ComplexArray,
    kinetic: FloatArray,
    potential: FloatArray,
    spacing: float,
    time_step: float = 0.02,
    num_steps: int = 1500,
    store_every: int = 25,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    """TODO: Apply potential half-step, FFT, kinetic phase, inverse FFT, potential half-step; save requested snapshots."""
    raise NotImplementedError


def periodic_fd_hamiltonian(
    potential: FloatArray, spacing: float, mass: float = 1.0, hbar: float = 1.0
) -> csr_matrix:
    """TODO: Build the three-point stencil plus the two periodic corner links in CSR form."""
    raise NotImplementedError


def propagate_cn(
    initial_states: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time_step: float = 0.01,
    num_steps: int = 3000,
    store_every: int = 50,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    """TODO: Factor the sparse CN left matrix once and reuse it for each update."""
    raise NotImplementedError


def solve_tunnelling(
    num_points: int = 1024,
    half_extent: float = 64.0,
    height: float = 2.5,
    width: float = 0.8,
    center: float = -20.0,
    sigma: float = 3.0,
    wave_number: float = 1.5,
    time_step: float = 0.02,
    num_steps: int = 1500,
    store_every: int = 25,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> TunnellingResult:
    """TODO: Prepare the grid, packet, and phases; propagate and package the snapshots."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the exercise; send long experiment output to a Markdown file."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
