# Code Quality

This project uses Ruff and mypy with Python 3.14.

## Install development tools

```powershell
uv sync
```

## Format code

```powershell
uv run ruff format .
```

This rewrites Python files into a consistent format.

## Check formatting without changing files

```powershell
uv run ruff format --check .
```

## Run the linter

```powershell
uv run ruff check .
```

Apply Ruff's safe automatic fixes with:

```powershell
uv run ruff check --fix .
```

Read each remaining message before changing the code. A remaining warning often
requires a design decision rather than a mechanical edit.

## Run the type checker

```powershell
uv run mypy .
```

Mypy is configured in strict mode. Function parameters and return values should
have explicit types.

## Run every check

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest
uv run python torch_smoke_test.py
```

## VS Code

Open the `ZeroToEndNumericalQM` directory as the workspace. VS Code will suggest
the Python, Ruff, and mypy extensions.

The workspace settings:

- select `.venv/Scripts/python.exe`;
- format Python on save;
- enable Ruff fixes and import sorting on explicit save actions;
- enable strict Python type analysis.

If diagnostics do not appear after installing the extensions, run
`Developer: Reload Window` from the Command Palette.

## Suppressing a warning

Do not disable a rule globally just to silence one line. First determine whether
the warning identifies a real problem. If an exception is justified, use the
smallest possible `# noqa: RULE` or `# type: ignore[error-code]` comment and add
a short reason.
