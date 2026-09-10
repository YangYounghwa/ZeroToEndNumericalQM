"""Starter: two coupled spins. Complete a copy in workbench/."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class CoupledEvolution:
    hamiltonians: ComplexArray
    times: FloatArray
    states: ComplexArray


def pauli_matrices() -> ComplexArray:
    """TODO: Build the three complex Pauli matrices in x,y,z order."""
    raise NotImplementedError


def local_operators() -> tuple[ComplexArray, ComplexArray]:
    """TODO: Use Kronecker products with identity to act on either site in the fixed basis."""
    raise NotImplementedError


def normalize_states(states: ComplexArray) -> ComplexArray:
    """TODO: Normalize the final four-amplitude axis and reject zero or nonfinite states."""
    raise NotImplementedError


def product_states(first: ComplexArray, second: ComplexArray) -> ComplexArray:
    """TODO: Form one 2x2 outer product per spinor pair, then flatten only the spin axes."""
    raise NotImplementedError


def hamiltonians(
    exchange: FloatArray, fields: FloatArray, gamma: float = 1.0, hbar: float = 1.0
) -> ComplexArray:
    """TODO: Combine J/4 times the three pair operators with the two signed Zeeman terms."""
    raise NotImplementedError


def evolve_constant(
    initial_states: ComplexArray,
    operators: ComplexArray,
    times: FloatArray,
    hbar: float = 1.0,
) -> CoupledEvolution:
    """TODO: Validate the batch and evaluate the static propagator at all requested times."""
    raise NotImplementedError


def propagate_cn(
    initial_states: ComplexArray,
    operators: ComplexArray,
    time_step: float = 0.1,
    num_steps: int = 100,
    store_every: int = 1,
    hbar: float = 1.0,
) -> CoupledEvolution:
    """TODO: Reuse the constant CN solve and save initial, final, and requested intermediate states."""
    raise NotImplementedError


def local_bloch_vectors(states: ComplexArray) -> FloatArray:
    """TODO: Contract joint states with each site's three Pauli operators."""
    raise NotImplementedError


def correlations(states: ComplexArray, connected: bool = False) -> FloatArray:
    """TODO: Compute all nine pair expectations, optionally subtracting products of local means."""
    raise NotImplementedError


def product_determinant(states: ComplexArray) -> FloatArray:
    """TODO: Evaluate the magnitude of the 2x2 coefficient determinant as a pure-state product test."""
    raise NotImplementedError


def energy_expectations(result: CoupledEvolution) -> FloatArray:
    """TODO: Contract each state with its own Hamiltonian, preserving time and batch axes."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the example or write a long experiment report to Markdown."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
