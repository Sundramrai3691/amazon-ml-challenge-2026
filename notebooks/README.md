# Notebooks

Production logic lives in `src/`. Notebooks must import that code rather than reimplement it.

Allowed uses:

- EDA
- visualizations
- rapid experiments
- comparison
- debugging

The first real notebook will be created separately (SageMaker Unified Studio): `00_problem_eda`.

Do not duplicate loaders, metrics, blocking, or submission writers in cells.
