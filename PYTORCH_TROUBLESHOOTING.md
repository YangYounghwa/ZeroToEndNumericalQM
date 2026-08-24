# PyTorch Troubleshooting

## Run the smoke test

From the repository root:

```powershell
uv run python torch_smoke_test.py
```

A successful CPU result is sufficient for the early chapters. CUDA is optional.

## `No module named 'torch'`

The project environment has not been synchronized, or the script is using a
different Python installation.

```powershell
uv sync
uv run python torch_smoke_test.py
```

Use `uv run`, not a separate global `python`, to select the project environment.

## PyTorch imports but CUDA is unavailable

Check the smoke-test output. A version ending in `+cpu` is CPU-only. This is not
an error for this project. Continue with `device="cpu"`.

If GPU execution is required:

1. Confirm that the machine has a supported NVIDIA GPU and a working driver.
2. Use the current installation command from the official PyTorch installation
   selector.
3. Re-run `torch_smoke_test.py`.

Do not install an arbitrary CUDA package version. The PyTorch build, GPU, and
driver must be compatible.

## `Torch not compiled with CUDA enabled`

The code selected `device="cuda"` while a CPU-only PyTorch build is installed.
Use safe device selection:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Install a compatible CUDA-enabled PyTorch build only if GPU execution is needed.

## Device mismatch

Errors mentioning different devices mean that some tensors are on CPU while
others are on CUDA. Create related tensors on one device:

```python
x = torch.arange(10, dtype=torch.float64, device=device)
identity = torch.eye(10, dtype=torch.float64, device=device)
```

## Dtype mismatch

Numerical quantum mechanics often needs `float64` or `complex128`. Specify the
dtype rather than relying on the PyTorch default:

```python
x = torch.arange(10, dtype=torch.float64)
```

All tensors participating in a matrix operation should normally use compatible
dtypes.

## DLL or import failure on Windows

First recreate only the project environment through `uv` rather than modifying
the global Python installation. If the error persists, record:

```powershell
uv run python --version
uv run python -c "import torch; print(torch.__version__)"
```

The exact DLL name and traceback are needed to distinguish an incompatible
wheel, missing runtime library, or damaged installation.

## `uv` cache failure

If the global cache path is damaged or inaccessible, use a project-local cache:

```powershell
uv --cache-dir .uv-cache sync
uv --cache-dir .uv-cache run python torch_smoke_test.py
```

