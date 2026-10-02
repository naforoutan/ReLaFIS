# ReLaFIS

### From Membership to Relations: A Neuro-Fuzzy Framework for Learning Relational Predicates

This repository contains the implementation and experimental code for **ReLaFIS (Relational Linguistic Neuro-Fuzzy Inference System)**, introduced in:

> **From Membership to Relations: A Neuro-Fuzzy Framework for Learning Relational Predicates**  
> Armin Salimi-Badr and Nazanin Foroutan  
> *Neurocomputing*, 2026

📄 **Paper:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231226026627)

---

## Overview

Traditional neuro-fuzzy systems typically construct rule antecedents using membership-based conditions such as:

> `x is A`

Although interpretable, this formulation mainly represents equality-like relations and limits the linguistic expressiveness of fuzzy rules.

**ReLaFIS** extends this formulation by introducing the **General Intuitive Fuzzy Term (GIFT)**, a differentiable fuzzy representation that allows a model to learn multiple relational interpretations of a linguistic term, including:

- approximately equal to
- not equal to
- greater than / at least
- less than / at most

Instead of fixing the relational meaning of an antecedent in advance, GIFT learns the appropriate relation directly from data.

ReLaFIS builds a complete neuro-fuzzy inference system around GIFT, enabling the construction of compact and linguistically richer fuzzy rules while maintaining end-to-end differentiability.

---

## Main Components

### General Intuitive Fuzzy Term (GIFT)

GIFT provides a unified representation for learning **equality, inequality, and order-based fuzzy relations**.

For each linguistic term, the model adaptively combines competing relational predicates rather than assuming only the conventional membership relation.

### Relational Fuzzy Antecedents

ReLaFIS learns fuzzy rules containing relational predicates such as:

```text
IF x₁ is approximately A
AND x₂ is at least B
AND x₃ is not C
THEN ...
```

This provides a richer linguistic representation than conventional neuro-fuzzy systems restricted primarily to `x is A` antecedents.

### Incomplete Fuzzy Rules

ReLaFIS incorporates an **indirect antecedent relaxation mechanism**.

When multiple relational alternatives receive comparable importance, the corresponding antecedent becomes less influential. This allows the model to naturally suppress unnecessary conditions and form **incomplete fuzzy rules** without requiring a separate feature-selection mechanism.

### Geometry-Aware Local Consequents

The consequent functions are defined in a rule-centered coordinate system and normalized according to the widths of the corresponding antecedent linguistic terms.

This couples local function approximation with the geometry of each learned fuzzy region.

### Reconstruction-Based Antecedent Learning

An auxiliary reconstruction objective is used during training to guide the antecedent representation.

The reconstruction branch operates directly on the normalized fuzzy-rule activations, encouraging the learned rule representation to preserve meaningful sample-dependent structure.

### Antecedent Relation Entropy (ARE)

The project also introduces **Antecedent Relation Entropy (ARE)**, an information-theoretic measure for quantifying the diversity of relational predicates expressed by a fuzzy rule base.

ARE can be used to analyze the linguistic richness of different neuro-fuzzy architectures.

---

## Model Architecture

At a high level, ReLaFIS follows the pipeline:

```text
Input
  │
  ▼
General Intuitive Fuzzy Terms (GIFT)
  │
  ├── Equality / Inequality Relations
  ├── Order-Based Relations
  └── Adaptive Relational Combination
  │
  ▼
Relational Rule Firing
  │
  ▼
Normalized Rule Activations
  │
  ├──────────────► Reconstruction Objective
  │
  ▼
Geometry-Aware Local TSK Consequents
  │
  ▼
Prediction
```

The complete architecture is implemented as a differentiable PyTorch model and can therefore be optimized end-to-end using gradient-based learning.

---

## Requirements

The implementation is based on Python and PyTorch.

Main dependencies include:

```text
torch
numpy
pandas
scikit-learn
scipy
matplotlib
```

Install the project dependencies using:

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Basic Usage

The main model is implemented as `ReLaFIS`.

A minimal initialization example is:

```python
from model.relafis import ReLaFIS

model = ReLaFIS(
    in_features=num_features,
    rules=num_rules,
    out_features=num_classes,
    binary=False,
)
```

For binary classification:

```python
model = ReLaFIS(
    in_features=num_features,
    rules=num_rules,
    out_features=1,
    binary=True,
)
```

The implementation also provides utilities for inspecting the learned relational structure and evaluating the linguistic properties of the resulting fuzzy rules.

---

## Experiments

The experimental evaluation considers several benchmark classification problems and compares ReLaFIS with conventional and recent neuro-fuzzy approaches.

The experiments investigate:

- predictive performance
- fuzzy-rule compactness
- relational linguistic richness
- antecedent relaxation
- sensitivity to the number of fuzzy rules
- robustness across independent runs
- alternative rule aggregation mechanisms
- behavior of learned relational predicates

The repository also contains the implementations and evaluation utilities required to reproduce the reported experiments.

---

## Interpretability

A central objective of ReLaFIS is to retain the interpretability of fuzzy inference while increasing its expressive power.

Rather than representing every antecedent only as:

```text
x is A
```

ReLaFIS can learn relational expressions such as:

```text
x is approximately A
x is not A
x is at least A
x is at most A
```

Different rules may therefore use different relational interpretations of the same input variable according to the local structure of the data.

---

## Related Models

ReLaFIS builds upon a line of work on interpretable neuro-fuzzy systems, including:

- **UNFIS** — learning incomplete fuzzy rules
- **LitANFIS** — learning positive and negative fuzzy literals
- **GRIFFIN** — gated, interaction-aware, incomplete and negation-aware fuzzy rules

ReLaFIS focuses specifically on extending fuzzy antecedents from conventional membership-based representations toward **learnable relational predicates**.

---

## Paper

If you use this repository in your research, please refer to:

**A. Salimi-Badr and N. Foroutan**,  
*"From Membership to Relations: A Neuro-Fuzzy Framework for Learning Relational Predicates,"*  
**Neurocomputing**, 2026.

[View the paper on ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231226026627)

---

## Citation

```bibtex
@article{salimibadr2026relafis,
  title   = {From Membership to Relations: A Neuro-Fuzzy Framework for Learning Relational Predicates},
  author  = {Salimi-Badr, Armin and Foroutan, Nazanin},
  journal = {Neurocomputing},
  year    = {2026},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0925231226026627}
}
```

The BibTeX entry can be updated with the final DOI, volume, pages/article number, and publication metadata once they are available.

---

## Contact

For questions regarding the paper or implementation, please contact the authors through the information provided in the publication.