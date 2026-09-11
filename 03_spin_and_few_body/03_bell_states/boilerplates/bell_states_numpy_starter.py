"""Starter: Bell states. Complete a copy in workbench/."""

import numpy as np
from numpy.typing import NDArray

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]


def pure_density(states: ComplexArray) -> ComplexArray:
    """TODO: Validate normalized kets and form conjugated outer products on the final axis."""
    raise NotImplementedError


def validate_density(rho: ComplexArray, *, tolerance: float = 1e-12) -> None:
    """TODO: Check square shape, finite entries, Hermiticity, unit trace, and positivity."""
    raise NotImplementedError


def mixed_density(states: ComplexArray, probabilities: RealArray) -> ComplexArray:
    """TODO: Average normalized projectors with matching nonnegative ensemble probabilities."""
    raise NotImplementedError


def partial_trace(
    rho: ComplexArray, dims: tuple[int, int] = (2, 2), *, keep: int = 0
) -> ComplexArray:
    """TODO: Reshape (a,b,a_prime,b_prime) and contract only the ignored subsystem."""
    raise NotImplementedError


def purity(rho: ComplexArray) -> RealArray:
    """TODO: Compute the trace of rho squared, preserving batch axes."""
    raise NotImplementedError


def entropy(rho: ComplexArray) -> RealArray:
    """TODO: Use Hermitian eigenvalues and the zero-log-zero limit, retaining tiny positive values."""
    raise NotImplementedError


def entanglement_entropy(
    states: ComplexArray, dims: tuple[int, int] = (2, 2)
) -> RealArray:
    """TODO: Reduce a PURE joint ket and calculate its local entropy in bits."""
    raise NotImplementedError


def bell_states() -> ComplexArray:
    """TODO: Construct Phi+, Phi-, Psi+, Psi- as normalized rows in the fixed basis."""
    raise NotImplementedError


def schmidt_states(angles: RealArray, *, phase: float = 0.0) -> ComplexArray:
    """TODO: Build cos(theta)|00> plus exp(i*phase)sin(theta)|11> for all angles."""
    raise NotImplementedError


def exchange_states(
    times: RealArray, *, exchange: float = 1.0, hbar: float = 1.0
) -> ComplexArray:
    """TODO: Use the exact zero-field exchange state from |01>, omitting global phase."""
    raise NotImplementedError


def pauli_correlations(rho: ComplexArray) -> RealArray:
    """TODO: Evaluate all nine expectations of sigma_a tensor sigma_b."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run a short example or write the reproducible report."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
