# Many-body exact diagonalization

Build small-system reference results before using approximate many-body
methods. PyTorch is the main learning tool. NumPy and sparse SciPy provide
independent checks.

## Available chapter

1. [Transverse-field Ising chain](01_transverse_field_ising/README.md)

The first chapter uses bit strings, open boundaries, global spin-flip parity,
magnetization, and pair correlations. It compares dense PyTorch with sparse
Kronecker-product construction and exact finite-chain benchmarks.

## Next subjects

2. Heisenberg spin chain: fixed-magnetization sectors and correlations.
3. Bose-Hubbard model: occupation bases and bosonic cutoff checks.
4. Fermi-Hubbard model: larger particle-number sectors and fermionic signs.

Increasing the chain length changes the physical model. Separate this
finite-size effect from iterative-solver error and basis truncation.
