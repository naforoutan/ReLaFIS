"""
Biomedical and clinical datasets.

This module is the natural home for datasets tied to health, biology, or
clinical research — including anything fetched live from the UCI ML Repository
via ucimlrepo.  Keeping them separate from the generic tabular modules makes
it easy to add genomics, EHR, or omics datasets later without cluttering the
general-purpose files.

Live fetch vs. local file
-------------------------
- *Live fetch* classes (e.g. Digits_UCI_Repo) pull data from the UCI API at
  runtime.  They require an internet connection and the `ucimlrepo` package.
- *Local file* classes (e.g. Diabetes) read from the project ./data/ tree.

To add a new dataset here:
  - Use ucimlrepo.fetch_ucirepo(id=<id>) for live fetch.
  - Set self.df = data.data.features and self.target = data.data.targets.
  - Call super().__init__() as usual.
"""

import pandas as pd
from .base import Test


class Diabetes(Test):
    """Diabetes dataset from a local CSV file (different from Pima / UCI versions).

    Expects ./data/diabetes/diabetes.csv with a header row; the last column
    is treated as the target by the parent class default.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("./data/diabetes/diabetes.csv")
        # Target column name is inferred by the Test base class from the last
        # column when self.target is not set explicitly — set it here if the
        # CSV layout differs from the default.
        super().__init__(*args, **kwargs)


class Digits_UCI_Repo(Test):
    """Optical Recognition of Handwritten Digits fetched live from UCI (id=81).

    Requires: pip install ucimlrepo
    """

    def __init__(self, *args, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=81)
        self.df = data.data.features
        self.target = data.data.targets
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Future slots — add below as you test new papers
# ---------------------------------------------------------------------------
# class EHRAdmissions(Test):   ...
# class GenomicsCancer(Test):  ...
# class Alzheimers(Test):      ...
