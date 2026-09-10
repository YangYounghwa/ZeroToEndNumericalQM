"""Starter: FFT tunnelling. Complete a copy in workbench/."""

from pathlib import Path


def time_step_study() -> list[str]:
    """TODO: Compare with a same-grid spectral matrix exponential and record error ratios."""
    raise NotImplementedError


def scattering_time_study() -> list[str]:
    """TODO: Refine dt at fixed physical setup; compare state, transmission, energy, and norm errors."""
    raise NotImplementedError


def grid_study() -> list[str]:
    """TODO: Refine the FFT grid and independently refine periodic finite differences."""
    raise NotImplementedError


def domain_study() -> list[str]:
    """TODO: Keep dx fixed; enlarge the box and compare common-grid states and edge histories."""
    raise NotImplementedError


def packet_width_study() -> list[str]:
    """TODO: Batch widths and compare incoming energy tails with outgoing region probabilities."""
    raise NotImplementedError


def measurement_study() -> list[str]:
    """TODO: Find a late transmission plateau before periodic wraparound."""
    raise NotImplementedError


def write_report(output: Path) -> None:
    """TODO: Collect study tables and write a Markdown report to the requested path."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the exercise; send long experiment output to a Markdown file."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
