"""Starter template for PyTorch harmonic-packet evolution."""

from torch import Tensor


def coherent_state(
    grid: Tensor,
    center: float = 2.0,
    wave_number: float = 0.7,
    angular_frequency: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    # TODO: Return a complex128 coherent-state packet.
    raise NotImplementedError


def propagate_crank_nicolson_batch(
    initial_states: Tensor,
    hamiltonian: Tensor,
    spacing: float,
    time_step: float,
    num_steps: int,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    # TODO: Propagate state columns with a reused LU factorization.
    raise NotImplementedError


def main() -> None:
    # TODO: Select a device and report physical checks.
    raise NotImplementedError


if __name__ == "__main__":
    main()
