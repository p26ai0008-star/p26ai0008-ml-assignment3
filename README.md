# ML Assignment 3 — CIFAR-10

Manish Yadav, roll number `p26ai0008`.
MLPs, CNNs, autoencoders, SVMs and clustering on CIFAR-10, all trained from scratch on CPU.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

torch is installed from the CPU wheel index (see requirements.txt header).

## Data

Download CIFAR-10 (`cifar-10-python.tar.gz`) into `data/`, then any script will
unpack and cache it as `data/cifar10_arrays.npz` on first use:

```bash
cd data && curl -O https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && cd ..
```

The train/validation split (seed = roll number digits, 260008) is stored in
`split/cifar10_split.npz`; the Q1b corrupted-label index file and the Q2b pixel
permutation are regenerated deterministically by their scripts if missing.
The 10,000-point Q4 subset (fixed before model selection, seed 260008) is
created by `q4_prepare.py` and stored in `data/q4_fixed.npz`.

## Reproduction commands

Run from the repository root, in this order. Every script resumes from
`checkpoints/` and appends to `logs/*.csv`, so runs can be interrupted and
repeated with `--seconds <budget>` slices without changing results.

```bash
python split_data.py
python q1a_train_mlps.py --model mlpL
python q1a_train_mlps.py --model mlpB
python q1a_train_mlps.py --model mlpS
python q1a_train_mlps.py --model mlpP
python q1b_memorisation.py
python q2a_train_cnn.py
python q2b_permutation.py --model mlpPerm
python q2b_permutation.py --model cnnPerm
python q2c_train_gray.py
python q2c_analysis.py
python q3a_autoencoders.py --bottleneck 4
python q3a_autoencoders.py --bottleneck 32
python q3a_autoencoders.py --bottleneck 128
python q3b_train_dae.py
python q3b_eval.py
python extract_features.py --run q3a_ae_c4
python extract_features.py --run q3a_ae_c32
python extract_features.py --run q3a_ae_c128
python extract_features.py --run q3b_dae_c128
python q3c_probes.py
python q4_prepare.py
python q4a_svm_sweep.py
python q4b_pairs.py
python q4c_clustering.py
python make_figures.py
```

`q4a_svm_sweep.py` accepts `--kernel {linear,rbf,poly}` and `--seconds` for
time-sliced execution; the linear kernel is solved on the one-vs-rest
squared-hinge primal with L-BFGS (see `q4a_linear_primal.py`), because
liblinear's dual hinge solver did not converge in reasonable time at large C.

The report PDF is built from `report/main.tex` (CVPR two-column style,
`report/cvpr.sty`) with `tectonic report/main.tex`.

## Artefacts

- `logs/*.csv` — per-epoch training curves for every run (Q1b also logs
  clean/noisy subset accuracy per epoch).
- `results/*.json` — final metrics, the full Q4a sweep, hard-pair analysis,
  clustering metrics; `results/q4a_sweep/*.json` — one file per SVM setting.
- `figures/*.pdf` — all report figures, produced by `make_figures.py`.
- `report/main.tex` — report source; numbers quoted there come from the JSON
  files above.
- `checkpoints/` and `data/` (feature caches, ~1 GB) are not committed; they
  are rebuilt by the commands above.

## Notes on protocol

- Validation only for all model/hyperparameter choices; the official test set
  is scored once per final model (Q1b never touches it).
- Q4 uses one fixed stratified subset of the 45,000 training codes
  (1,000 per class, seed 260008), created before any SVM tuning and used for
  every sweep row; k-means uses the same scaled codes.
- Clustering model and k are chosen with training and validation data only.
- The commit hash on page 1 of the report refers to the code commit (this
  repository state without the report tweak); the report source is committed
  right after it.
