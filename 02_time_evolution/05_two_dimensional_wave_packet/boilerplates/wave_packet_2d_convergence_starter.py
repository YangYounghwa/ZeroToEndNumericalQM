"""Starter: two-dimensional wave packets. Complete a copy in workbench/."""

from pathlib import Path

import torch
import wave_packet_2d_torch_solution as solution


def exact_coherent_state(result: solution.Evolution2D) -> torch.Tensor:
    """TODO: Construct the final analytical displaced Gaussian with continuum normalization, allowing a global phase."""
    raise NotImplementedError


def spectral_time_study() -> list[str]:
    """TODO: Hold the spatial matrix fixed and compare with its exponential as dt decreases."""
    raise NotImplementedError


def coupled_time_study() -> list[str]:
    """TODO: Compare states, centers, covariance, and energy with normal-mode references while refining dt."""
    raise NotImplementedError


def grid_study() -> list[str]:
    """TODO: Refine Nx and Ny at fixed box and dt; compare common-grid states and NumPy."""
    raise NotImplementedError


def domain_study() -> list[str]:
    """TODO: Hold dx and dy fixed while enlarging the box; compare common-grid states and edge histories."""
    raise NotImplementedError


def free_packet_study() -> list[str]:
    """TODO: Compare free Gaussian centers, variances, and states with analytical dynamics."""
    raise NotImplementedError


def write_report(output: Path) -> None:
    """TODO: Collect numerical tables and write the requested Markdown report."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the exercise and write long experiment output to Markdown."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
