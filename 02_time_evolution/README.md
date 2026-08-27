# Time Evolution

This phase studies the time-dependent Schrödinger equation with complex
wavefunctions and unitary numerical propagation.

## Chapters

1. [Free Gaussian wave packet](01_free_gaussian_wave_packet/README.md)
2. [Wave packet in a harmonic potential](02_harmonic_wave_packet/README.md)

## Current methods

- Small dense matrix exponential as a reference
- Sparse Crank-Nicolson propagation
- Reused LU factorizations
- Batched PyTorch propagation
- Norm, energy, reversibility, and time-step checks

## Next chapter

Scattering from a potential barrier will introduce reflected and transmitted
probabilities and require careful separation of the outgoing packets.
