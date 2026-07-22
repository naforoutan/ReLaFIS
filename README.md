# ReLaFIS

PyTorch implementation of **ReLaFIS** (Relational Linguistic Fuzzy Inference System), a neuro-fuzzy classifier with dual-branch relational antecedents (equality and ordering), soft combination weights, and entropy-aware consequent relaxation.

## Features

- Dual-branch antecedents: equal / not-equal and at-least / at-most
- Final effective membership via a learned mix of both branches
- TSK consequents with per-feature relaxation
- Absolute relational entropy (ARE) over relation categories
- scikit-learn-compatible wrapper for training and evaluation
- Tabular / biomedical / UCI dataset loaders with holdout or stratified k-fold splits

## Repository layout

```
model/ReLaFIS.py       # ReLaFIS, MamdaniReLaFIS, SklearnReLaFISWrapper
tests/                 # Dataset / experiment base classes
train/                 # Early stopping and training helpers
notebooks/test_class.ipynb
```

## Quick start

```python
import torch
from model.ReLaFIS import ReLaFIS, SklearnReLaFISWrapper

model = ReLaFIS(
    in_features=X.shape[1],
    rules=3,
    out_features=1,      # binary
    binary=True,
)
wrapper = SklearnReLaFISWrapper(model, device="cuda" if torch.cuda.is_available() else "cpu")
wrapper.fit(X_train, y_train)   # or train with your own loop
y_pred = wrapper.predict(X_test)
```

Interactive experiments and dataset setup live in `notebooks/test_class.ipynb`.

## Metrics

| Metric | Meaning |
|--------|---------|
| **ARE (linguistic richness)** | Mean Shannon entropy of hard-assigned relation categories per rule; range \([0,\ \ln 4]\) nats |
| **Relaxation rate** | Mean semantic relaxation \(\rho \in [0,1]\); higher ⇒ more “don’t care” attenuation |

## Requirements

- Python 3.10+
- PyTorch, NumPy, scikit-learn, pandas
- PyCaret (dataset / split utilities in `tests/`)

## Citation

If you use this code, please cite the associated ReLaFIS paper (update with the final venue reference when available).
