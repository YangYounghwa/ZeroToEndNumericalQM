"""Small environment test for the PyTorch features used by this project."""

import sys
import traceback


def run_cpu_checks() -> None:
    import torch

    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert values.dtype == torch.float64
    assert torch.dot(values, values).item() == 14.0

    matrix = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    expected = torch.tensor([1.0, 3.0], dtype=torch.float64)
    assert torch.allclose(eigenvalues, expected)
    assert torch.allclose(
        eigenvectors.mT @ eigenvectors, torch.eye(2, dtype=torch.float64)
    )

    parameter = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    loss = parameter**2
    # PyTorch's Tensor stub does not type this runtime-supported method.
    loss.backward()  # type: ignore[no-untyped-call]
    assert parameter.grad is not None
    assert parameter.grad.item() == 6.0


def run_cuda_check() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False

    matrix = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float64, device="cuda")
    eigenvalues = torch.linalg.eigvalsh(matrix)
    torch.cuda.synchronize()
    expected = torch.tensor([1.0, 3.0], dtype=torch.float64, device="cuda")
    assert torch.allclose(eigenvalues, expected)
    return True


def main() -> int:
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"Python version: {sys.version.split()[0]}")
        print(f"PyTorch CUDA build: {torch.version.cuda}")

        run_cpu_checks()
        print("PASS: CPU tensors, float64, eigensolver, and autograd")

        if run_cuda_check():
            print(f"PASS: CUDA on {torch.cuda.get_device_name(0)}")
        else:
            print("SKIP: CUDA is unavailable; CPU calculations remain supported")

        return 0
    except ImportError:
        print("FAIL: PyTorch cannot be imported")
        print("FIX: run `uv sync`, then execute this script with `uv run`.")
    except Exception:
        print("FAIL: a PyTorch operation failed")
        traceback.print_exc()
        print("FIX: see PYTORCH_TROUBLESHOOTING.md.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
