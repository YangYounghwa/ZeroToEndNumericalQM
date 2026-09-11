"""Starter: Bell states. Complete a copy in workbench/."""

from pathlib import Path


def binary_entropy(probability: float) -> float:
    """TODO: Evaluate the scalar binary entropy, including probabilities zero and one."""
    raise NotImplementedError


def state_study() -> list[str]:
    """TODO: Compare joint and reduced purities, entropies, and Bell/classical correlations."""
    raise NotImplementedError


def schmidt_study() -> list[str]:
    """TODO: Sweep the Schmidt angle and compare reduced entropy with its scalar formula."""
    raise NotImplementedError


def exchange_study() -> list[str]:
    """TODO: Track entanglement through an exact swap and compare with matrix evolution."""
    raise NotImplementedError


def precision_study() -> list[str]:
    """TODO: Check zero and tiny positive eigenvalues; explain the absence of dt/dx refinement."""
    raise NotImplementedError


def library_study() -> list[str]:
    """TODO: Compare seeded batches with the independent NumPy index-loop implementation."""
    raise NotImplementedError


def write_report(path: Path) -> None:
    """TODO: Write physical parameter and numerical precision tables to Markdown."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run a short example or write the reproducible report."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
