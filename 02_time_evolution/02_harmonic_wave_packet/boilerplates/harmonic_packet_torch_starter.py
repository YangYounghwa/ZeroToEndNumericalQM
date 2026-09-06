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


def matrix_exponential_state(
    initial_state: Tensor,
    hamiltonian: Tensor,
    spacing: float,
    time: float,
    hbar: float = 1.0,
) -> Tensor:
    # TODO: Normalize, then apply torch.linalg.matrix_exp(-1j * time * H / hbar).
    # Keep this dense reference small and retain the state's device.
    raise NotImplementedError


def state_l2_error(numerical: Tensor, reference: Tensor, spacing: float) -> Tensor:
    # TODO: Align the global phase using the weighted conjugate overlap.
    # Return the weighted L2 norm of the difference.
    raise NotImplementedError


def main() -> None:
    # TODO: Select a device and report physical checks.
    raise NotImplementedError


if __name__ == "__main__":
    main()
