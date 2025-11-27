
# Homomorphic Encryption + Federated Learning (HE + FL)

This repository demonstrates privacy-preserving federated learning workflows combined with homomorphic encryption (HE) to enable collaborative model training on sensitive video and image datasets without exposing raw data.

Overview
-
- Purpose: Provide reproducible examples and reference implementations for research and engineering around HE-enabled federated learning on small-to-medium scale datasets (MedMNIST and video subsets).
- Contents: notebooks, client/server training orchestration, example encrypted model artifacts, and helper scripts for running locally (macOS) or in Docker.

Highlights
-
- Federated learning agent implementations (RL-based agents and classical ML baselines).
- Integration with Pyfhel for homomorphic encryption demonstrations (macOS shim + Docker-ready real HE flow).
- Example Graph Neural Network (GNN) client modeling for relationship-aware aggregation.

Repository Structure
-
- `repo_HE_FL/` — Core federated-learning implementation, models, and helper scripts.
- `data/` — Example datasets (do not commit large datasets; use local or cloud storage in production).
- `weights/` — Example model weights (ignored by `.gitignore`).
- `*.ipynb` — Analysis and runnable notebooks demonstrating experiments.

Quick Start
-
Prerequisites
-
- Python 3.9+ (3.10 recommended)
- `pip` or `conda`
- (Optional) GPU and corresponding CUDA toolkit for accelerated PyTorch training.

Install (recommended: virtual environment)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run Notebooks
-
- Launch Jupyter Lab / Notebook and open the example notebooks:

```bash
jupyter lab
```

- The primary notebooks are:
	- `rl-fl-ensemble (1).ipynb` — RL + Federated learning ensemble experiments
	- `Encrypted FL Main-Rel.ipynb` — Main pipeline mixing HE and FL

Development
-
- Use the included `.github/workflows/python-app.yml` to run lightweight CI (lint/test) on GitHub Actions.

Best Practices and Notes
-
- Do NOT store private datasets or large model artifacts in the repo. Use `.gitignore` (provided) and external storage for datasets and trained weights.
- If you need to share artifacts for review, provide small reproducible subsets or scripts to fetch data from a private bucket.

Contributing
-
See `CONTRIBUTING.md` for contribution guidelines, code style, and how to run tests locally.

License
-
This project includes a `LICENSE` file at the repository root. Ensure compatibility before reuse.

Acknowledgements
-
- Research and demo code adapted from internal experiments and open-source references in federated learning, PyTorch, and Pyfhel.


