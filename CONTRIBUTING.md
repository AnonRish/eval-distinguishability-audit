# Contributing

1. `python -m pip install -e ".[dev]"`
2. `ruff check . && ruff format --check . && mypy src && pytest`
3. Every behavioural claim in `docs/ASSURANCE.md` must map to a test. If you add a
   claim, add the test; if you change behaviour, update the claim.
4. CI enforces 100% line and branch coverage. Do not add `# pragma: no cover`
   without a written justification in the pull request.

Please keep the scope honest: this project reports what its tests establish and
says plainly what they do not.
