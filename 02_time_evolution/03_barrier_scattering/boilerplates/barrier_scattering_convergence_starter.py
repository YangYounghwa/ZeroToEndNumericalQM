"""Starter exercise: rectangular-barrier packet scattering. Complete a copy in workbench/."""

import torch


def state_error(first: torch.Tensor, second: torch.Tensor, spacing: float) -> float:
    """TODO: Align a global phase from the weighted overlap before calculating L2 error."""
    raise NotImplementedError


def time_step_study() -> list[str]:
    """TODO: Fix H and the final time; compare PyTorch CN against exponential references as dt decreases."""
    raise NotImplementedError


def grid_study() -> list[str]:
    """TODO: Fix physical parameters and align barrier edges during refinement; compare with spectrum-averaged T."""
    raise NotImplementedError


def measurement_time_study() -> list[str]:
    """TODO: Record left/near/right and edge history; find a post-collision plateau before wall returns."""
    raise NotImplementedError


def domain_study() -> list[str]:
    """TODO: Keep dx fixed while enlarging the walls; compare states on the common grid and edge histories."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the chapter exercise and save long experiment results in Markdown."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
