#!/usr/bin/env bash

# RL + Federated Learning Ensemble (rl-fl-ensemble)

This repository contains experimental notebooks and helper code for combining Reinforcement Learning (RL) with Federated Learning (FL) in video recognition experiments. The main working artifact is the notebook `rl-fl-ensemble (1).ipynb`.

This project mixes a few research directions:
- Federated client modeling and aggregation
- Reinforcement-learning-driven aggregation/ensemble strategies
- (Optional) Homomorphic encryption integration for privacy-preserving aggregation (see `repo_HE_FL/`)

Key paths
- `rl-fl-ensemble (1).ipynb` — Primary notebook driving experiments and visualizations
- `repo_HE_FL/` — helper code, example weights, and HE-related utilities
- `data/` — datasets (not included in repo; add locally or configure paths in the notebook)

Overview
--------
The notebook is designed to be runnable end-to-end in a reproducible environment. It contains cells to prepare data splits, construct client datasets, define local models (MLP / CNN / GNN), perform local training, and implement aggregation strategies including an RL-based ensemble.

Quickstart
----------
1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Launch Jupyter Lab/Notebook and open `rl-fl-ensemble (1).ipynb`:

```bash
jupyter lab
# or
jupyter notebook
```

3. Run cells sequentially. The notebook includes a helper cell that attempts to auto-install `torch_geometric` if it is missing.

Notes on GPU/CPU
-----------------
- For GPU support install matching `torch` wheels from https://pytorch.org. The included `requirements.txt` is CPU-friendly; change to CUDA-specific wheels for GPU experiments.

Data and Large Files
--------------------
This repository does not include large datasets or some model weight files. Add your local datasets to `data/` or configure the paths in the notebook. Example weights may be present in `repo_HE_FL/weights/` but large binaries are intentionally not tracked.

Contributing
------------
See `CONTRIBUTING.md` for contribution guidelines and `CODE_OF_CONDUCT.md` for community standards.

CI
--
A GitHub Actions workflow is provided in `.github/workflows/python-app.yml` to run basic linting and tests (if present).

License
-------
This repository includes a `LICENSE` file. Review and replace if you prefer a different license.

Contact
-------
Open an issue or PR on the repository once pushed to GitHub for questions or collaboration.


