# Copilot instructions

- Never add AI attribution to commits or PRs: no "Co-Authored-By: Claude/Copilot",
  no "Generated with ..." footers. Commits carry the author's identity only.
- Conventional Commits, type set: feat / fix / deps / chore / docs. One PR = one type.
- Toolchain: uv, ruff (E,F,I,UP,B,D + NumPy docstrings), strict mypy, pytest
  (`bin/lint`, `bin/test`). Match the surrounding code's style and comment density.
- Surgical changes only: every changed line traces to the request; don't reformat or
  "improve" adjacent code.
- Never put credentials in code or configs; `.env` stays gitignored.
