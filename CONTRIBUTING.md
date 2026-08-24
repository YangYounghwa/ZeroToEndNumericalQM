# Contributing

Keep commits and pull requests small, clear, and focused on one purpose.

## Commit messages

Use this format:

```text
type: short summary
```

Allowed types:

- `feat`: add new material or behavior
- `fix`: correct an error
- `docs`: change documentation only
- `test`: add or update tests
- `refactor`: improve code without changing behavior
- `chore`: maintenance or configuration

Rules:

- Write the summary as a command, such as `add`, `fix`, or `update`.
- Keep the first line under 72 characters.
- Do not end the summary with a period.
- Add a body only when the reason or an important decision is not obvious.

Examples:

```text
feat: add finite square well solver
fix: correct wavefunction normalization
docs: explain convergence criteria
```

## Pull requests

Use the same format for the PR title:

```text
type: short summary
```

The PR description must state:

1. Why the change is needed.
2. What changed.
3. How it was verified.

Link an issue when one exists. Keep each PR focused on one topic.

## Branches

Direct work on `main` is acceptable while the project has one contributor.
When branches are used, name them with this format:

```text
type/short-description
```

Examples: `feat/finite-well`, `fix/normalization`, `docs/setup-guide`.

