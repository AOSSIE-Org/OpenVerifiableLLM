# Developer Setup

This project uses `uv` for dependency management.

## Installation
1. Install [uv](https://github.com/astral-sh/uv).
2. Run `uv sync` to install dependencies and create a virtual environment.

## Linting
We use `ruff` to maintain code quality. 
- Run checks: `uv run ruff check .`
- Auto-fix issues: `uv run ruff check . --fix`