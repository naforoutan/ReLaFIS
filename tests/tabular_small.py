"""
Small and medium classic tabular classification datasets.

All real-world datasets are fetched live from the UCI Machine Learning
Repository (via ``ucimlrepo`` or direct UCI file URLs) or from OpenML
(via ``sklearn.datasets.fetch_openml``). No local ``./data/`` files are
required.

Requires an internet connection on first fetch (results are cached by the
respective libraries). ``ucimlrepo`` is needed for UCI API imports
(``pip install ucimlrepo``).

To add a new dataset here:
  1. Subclass Test
  2. Fetch via ``_fetch_uci`` / ``_fetch_openml`` / ``_fetch_uci_excel``
  3. Set self.df (features) and self.target (Series), then call super()
"""

import io
import urllib.request

import numpy as np
import pandas as pd
from .base import Test


# ---------------------------------------------------------------------------
# Shared fetch helpers
# ---------------------------------------------------------------------------

def _unique_columns(columns):
    """Make column labels unique (UCI sometimes truncates names)."""
    seen = {}
    out = []
    for col in columns:
        col = str(col)
        if col not in seen:
            seen[col] = 0
            out.append(col)
        else:
            seen[col] += 1
            out.append(f"{col}_{seen[col]}")
    return out


def _prepare(features, target, target_name="class"):
    """Normalize feature/target frames returned by UCI or OpenML."""
    df = features.copy()
    if isinstance(target, pd.DataFrame):
        target = target.iloc[:, 0]
    target = pd.Series(target).reset_index(drop=True)
    df = df.reset_index(drop=True)
    df.columns = _unique_columns(df.columns)
    target.name = target_name
    return df, target


def _fetch_uci(uci_id: int):
    """Fetch features/targets from the UCI API via ucimlrepo."""
    from ucimlrepo import fetch_ucirepo

    data = fetch_ucirepo(id=uci_id)
    return data.data.features.copy(), data.data.targets


def _fetch_openml(name=None, data_id=None, version=1, target_name="class"):
    """Fetch a dataset from OpenML via scikit-learn."""
    from sklearn.datasets import fetch_openml

    kwargs = {"as_frame": True, "parser": "auto"}
    if data_id is not None:
        data = fetch_openml(data_id=data_id, **kwargs)
    else:
        data = fetch_openml(name, version=version, **kwargs)

    features = data.data.copy()
    target = data.target
    if target is None and hasattr(data, "frame") and data.frame is not None:
        # Some OpenML dumps put the label in the full frame only.
        raise ValueError(f"OpenML dataset {name or data_id} has no target column")
    return _prepare(features, target, target_name=target_name)


def _fetch_uci_excel(url: str) -> pd.DataFrame:
    """Download an Excel file hosted on the UCI ML Repository."""
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()
    return pd.read_excel(io.BytesIO(raw))


# ---------------------------------------------------------------------------
# sklearn / UCI classics
# ---------------------------------------------------------------------------

class Iris(Test):
    """Iris plants — UCI id=53 (150 samples, 4 features, 3 classes)."""

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(53)
        self.df, self.target = _prepare(features, targets, target_name="class")
        super().__init__(*args, **kwargs)


class Digits(Test):
    """Sklearn 8×8 handwritten digits (UCI optical-digits derivative).

    1 797 samples, 64 features, 10 classes. Kept via sklearn because this
    is the standard 8×8 preprocessing used in the literature; the full
    UCI optical-digits table (5 620 samples) is ``DigitsUCI``.
    """

    def __init__(self, *args, **kwargs) -> None:
        from sklearn import datasets

        data = datasets.load_digits(as_frame=True)
        self.df = data.data.copy()
        self.df.columns = self.df.columns.astype(str)
        self.target = data.target.copy()
        self.target.name = "digit"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Medical / treatment
# ---------------------------------------------------------------------------

class Cryotheraphy(Test):
    """Cryotherapy wart-treatment outcome — UCI id=429 (Excel download).

    90 samples, 6 features, binary ``Result_of_Treatment``. Not currently
    importable via the UCI API, so the Excel file is fetched from the
    UCI machine-learning-databases mirror.
    """

    _URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00429/Cryotherapy.xlsx"
    )

    def __init__(self, *args, **kwargs) -> None:
        data = _fetch_uci_excel(self._URL)
        self.target = data["Result_of_Treatment"].astype(int)
        self.df = data.drop(columns=["Result_of_Treatment"])
        self.df, self.target = _prepare(self.df, self.target, target_name="Result_of_Treatment")
        super().__init__(*args, **kwargs)


class Immunotherapy(Test):
    """Immunotherapy wart-treatment outcome — UCI id=428 (Excel download).

    90 samples, 7 features, binary ``Result_of_Treatment``. Same fetch
    pattern as ``Cryotheraphy`` (UCI Excel mirror).
    """

    _URL = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "00428/Immunotherapy.xlsx"
    )

    def __init__(self, *args, **kwargs) -> None:
        data = _fetch_uci_excel(self._URL)
        self.target = data["Result_of_Treatment"].astype(int)
        self.df = data.drop(columns=["Result_of_Treatment"])
        self.df, self.target = _prepare(self.df, self.target, target_name="Result_of_Treatment")
        super().__init__(*args, **kwargs)


def _fetch_pima_online():
    """Fetch the UCI Pima Indians Diabetes dataset from OpenML (id=37).

    The classic Pima dataset was removed from the UCI API, but OpenML mirrors
    the identical 768-sample / 8-feature version. Returns ``(features_df,
    target_series)`` with the target mapped to 1 (positive) / 0 (negative).
    """
    features, target = _fetch_openml("diabetes", version=1, target_name="class")
    target = target.astype(str).map(
        {"tested_positive": 1, "tested_negative": 0}
    )
    target.name = "class"
    return features, target


class PimaDiabetes(Test):
    """Pima Indians Diabetes — Case 1 (OpenML ``diabetes``, UCI mirror).

    Full Pima set: all 768 samples, 8 features, binary target. Medically
    impossible zero values (e.g. glucose/BMI = 0) are kept as-is.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.df, self.target = _fetch_pima_online()
        super().__init__(*args, **kwargs)


class PimaDiabetesCase2(Test):
    """Pima Indians Diabetes — Case 2 (OpenML ``diabetes``, UCI mirror).

    Rows with medically impossible zeros are removed in any of:
    plasma glucose (``plas``), diastolic blood pressure (``pres``), triceps
    skin-fold thickness (``skin``), 2-hour serum insulin (``insu``), or BMI
    (``mass``). ~392 complete samples remain.
    """

    _IMPOSSIBLE_ZERO_COLS = ["plas", "pres", "skin", "insu", "mass"]

    def __init__(self, *args, **kwargs) -> None:
        df, target = _fetch_pima_online()

        before = len(df)
        mask = (df[self._IMPOSSIBLE_ZERO_COLS] == 0).any(axis=1)
        self.df = df[~mask].reset_index(drop=True)
        self.target = target[~mask].reset_index(drop=True)
        if len(self.df) < before:
            print(
                f"PimaDiabetesCase2: dropped {before - len(self.df)} rows "
                f"with impossible zeros ({len(self.df)} samples remain)."
            )

        super().__init__(*args, **kwargs)


class BCW(Test):
    """Breast Cancer Wisconsin (Original) — UCI id=15.

    699 samples, 9 cytological features, binary Class (2 = benign,
    4 = malignant). ``Bare_nuclei`` has a few missing values handled by
    the shared imputation pipeline.
    """

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(15)
        self.df, self.target = _prepare(features, targets, target_name="Class")
        super().__init__(*args, **kwargs)


class BreastCancer(Test):
    """Breast Cancer Wisconsin (Diagnostic / WDBC) — UCI id=17.

    569 samples, 30 features, binary Diagnosis (M/B → 1/0).
    """

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(17)
        df, target = _prepare(features, targets, target_name="Diagnosis")
        self.target = target.astype(str).str.upper().map({"M": 1, "B": 0}).astype(int)
        self.target.name = "Diagnosis"
        self.df = df
        super().__init__(*args, **kwargs)


class Autism(Test):
    """Autism Screening Adult — UCI id=426.

    704 samples, binary ASD vs non-ASD. Drops ``result`` (aggregate
    screening score — perfect proxy for the label) and ``age_desc``
    (redundant with ``age``). Uses simple imputation because only ``age``
    has missing values.
    """

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(426)
        df, target = _prepare(features, targets, target_name="ASD")

        drop_cols = [c for c in ("result", "age_desc") if c in df.columns]
        self.df = df.drop(columns=drop_cols)
        self.target = target.astype(str).str.upper().map({"YES": 1, "NO": 0}).astype(int)
        self.target.name = "ASD"

        kwargs.setdefault("imputation_type", "simple")
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Physical / natural science
# ---------------------------------------------------------------------------

class Glass(Test):
    """Glass Identification — UCI id=42 (214 samples, 9 features, 6 classes)."""

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(42)
        self.df, self.target = _prepare(features, targets, target_name="Type_of_glass")
        super().__init__(*args, **kwargs)


class Wine(Test):
    """Wine recognition — UCI id=109 (178 samples, 13 features, 3 classes)."""

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(109)
        self.df, self.target = _prepare(features, targets, target_name="class")
        super().__init__(*args, **kwargs)


class Thyroid(Test):
    """New Thyroid — OpenML ``thyroid-new`` (data_id=40682).

    Classic 215-sample / 5-feature / 3-class thyroid dataset (UCI
    ``new-thyroid``). The full Thyroid Disease suite is not importable
    via the UCI API, so this OpenML mirror is used instead.
    """

    _FEATURE_NAMES = (
        "T3_resin",
        "thyroxin",
        "triiodothyronine",
        "TSH",
        "TSH_diff",
    )

    def __init__(self, *args, **kwargs) -> None:
        self.df, self.target = _fetch_openml(
            data_id=40682, target_name="class"
        )
        if self.df.shape[1] == len(self._FEATURE_NAMES):
            self.df.columns = list(self._FEATURE_NAMES)
        self.target = pd.to_numeric(self.target, errors="raise").astype(int)
        self.target.name = "class"
        super().__init__(*args, **kwargs)


class Parkinson(Test):
    """Parkinsons (Oxford) — UCI id=174.

    195 voice recordings, 22 biomedical voice measures, binary status
    (1 = Parkinson's, 0 = healthy).
    """

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(174)
        self.df, self.target = _prepare(features, targets, target_name="status")
        self.target = pd.to_numeric(self.target, errors="raise").astype(int)
        self.target.name = "status"

        print(f"Parkinson: {len(self.df)} samples, {self.df.shape[1]} features")
        print(f"Class distribution:\n{self.target.value_counts().sort_index()}")

        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Cardiovascular / survival
# ---------------------------------------------------------------------------

class Heart(Test):
    """Cleveland Heart Disease — UCI id=45.

    303 samples, 13 features. Target ``num`` (0–4 angiographic status) is
    binarised to disease present (``num > 0``) vs absent.
    """

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(45)
        df, target = _prepare(features, targets, target_name="num")
        # Drop rows with missing target (none today) and keep feature NaNs
        # for the shared imputation pipeline (ca / thal historically have ?).
        mask = target.notna()
        self.df = df.loc[mask].reset_index(drop=True)
        self.target = (pd.to_numeric(target.loc[mask], errors="coerce") > 0).astype(int)
        self.target.name = "heart_disease"
        self.target = self.target.reset_index(drop=True)
        super().__init__(*args, **kwargs)


class Haberman(Test):
    """Haberman's Survival — UCI id=43 (306 samples, 3 features, binary)."""

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(43)
        df, target = _prepare(features, targets, target_name="survival_status")
        combined = pd.concat([df, target], axis=1)
        before = len(combined)
        combined = combined.dropna(subset=[target.name])
        if len(combined) < before:
            print(f"Haberman: dropped {before - len(combined)} rows with NaNs.")
        self.df = combined.iloc[:, :-1]
        self.target = combined.iloc[:, -1]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "survival_status"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Image / digit (tabular representation)
# ---------------------------------------------------------------------------

class DigitsUCI(Test):
    """Optical Recognition of Handwritten Digits — UCI id=80.

    Combined train+test tables: 5 620 samples, 64 pixel features, 10 classes.
    """

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(80)
        self.df, self.target = _prepare(features, targets, target_name="digit")
        self.target = pd.to_numeric(self.target, errors="raise").astype(int)
        self.target.name = "digit"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Online loaders — text / sequence
# ---------------------------------------------------------------------------

class DNA(Test):
    """StatLog DNA — OpenML ``dna`` (splice-junction gene sequences).

    3 186 samples, 180 binary features (60 nucleotide positions one-hot
    encoded into 3 indicators each) and a 3-class target.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.df, self.target = _fetch_openml("dna", version=1, target_name="class")
        self.df = self.df.apply(pd.to_numeric).astype(int)
        self.target = self.target.astype(int)
        self.target.name = "class"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Multi-view / high-dimensional
# ---------------------------------------------------------------------------

class MFeat(Test):
    """Multiple Features (MFeat) — UCI id=72, OpenML fallback.

    Six handwritten-digit feature views concatenated: 2 000 samples,
    649 features, 10 classes. Tries UCI first; falls back to the six
    OpenML views when the UCI API does not expose this dataset.
    """

    _OPENML_VIEWS = (
        ("factors", "mfeat-factors"),
        ("fourier", "mfeat-fourier"),
        ("karhunen", "mfeat-karhunen"),
        ("morph", "mfeat-morphological"),
        ("pixel", "mfeat-pixel"),
        ("zernike", "mfeat-zernike"),
    )

    @classmethod
    def _load_from_openml(cls):
        from sklearn.datasets import fetch_openml

        frames = []
        target = None
        for short, name in cls._OPENML_VIEWS:
            data = fetch_openml(name, version=1, as_frame=True, parser="auto")
            part = data.data.copy()
            part.columns = [f"{short}_{i}" for i in range(part.shape[1])]
            frames.append(part)
            if target is None:
                target = data.target

        return pd.concat(frames, axis=1), target

    def __init__(self, *args, **kwargs) -> None:
        try:
            features, targets = _fetch_uci(72)
            self.df, self.target = _prepare(features, targets, target_name="digit")
        except Exception:
            features, target = self._load_from_openml()
            self.df, self.target = _prepare(features, target, target_name="digit")

        self.target = pd.to_numeric(self.target, errors="raise").astype(int)
        self.target.name = "digit"

        print(
            f"✅ MFeat loaded: {self.df.shape[0]} samples, "
            f"{self.df.shape[1]} features, {self.target.nunique()} classes"
        )
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# UCI fetch — image segmentation (tabular region features)
# ---------------------------------------------------------------------------

class SegmentaitionUCI(Test):
    """Image Segmentation — UCI id=50.

    Each row is a 3×3 image region with 19 texture/color features and a
    material class label (7 classes). Note: the UCI API currently exposes
    the 210-row training split.
    """

    def __init__(self, *args, **kwargs) -> None:
        features, targets = _fetch_uci(50)
        df, target = _prepare(features, targets, target_name="class")
        combined = pd.concat([df, target], axis=1)
        before = len(combined)
        combined = combined.dropna(subset=[target.name])
        if len(combined) < before:
            print(f"Segmentation: dropped {before - len(combined)} rows with NaNs.")

        self.df = combined.iloc[:, :-1]
        self.target = combined.iloc[:, -1]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "class"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Categorical / ordinal
# ---------------------------------------------------------------------------

class CarEvaluation(Test):
    """Car Evaluation — UCI id=19 (1 728 samples, 6 ordinal features, 4 classes)."""

    def __init__(self, *args, **kwargs) -> None:
        from sklearn.preprocessing import LabelEncoder

        features, targets = _fetch_uci(19)
        df, target = _prepare(features, targets, target_name="class")
        le = LabelEncoder()
        self.target = pd.Series(le.fit_transform(target.astype(str)), name="class")
        self.df = df
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Synthetic / generative
# ---------------------------------------------------------------------------

class SyntheticGaussian(Test):
    """Synthetic 2-D Gaussian clusters for sanity-checking classifiers."""

    def __init__(self, n_clusters: int = 5, n_samples: int = 1000, *args, **kwargs) -> None:
        np.random.seed(42)
        centers = np.random.randn(n_clusters, 2) * 3
        spc = n_samples // n_clusters
        X, y = [], []
        for i, c in enumerate(centers):
            X.extend(np.random.randn(spc, 2) * 0.5 + c)
            y.extend([i] * spc)
        remaining = n_samples - len(X)
        if remaining > 0:
            X.extend(np.random.randn(remaining, 2) * 0.5 + centers[-1])
            y.extend([n_clusters - 1] * remaining)

        X, y = np.array(X), np.array(y)
        idx = np.random.permutation(len(X))
        self.df = pd.DataFrame(X[idx], columns=["x", "y"])
        self.target = pd.Series(y[idx], name="cluster")
        self.df.columns = self.df.columns.astype(str)

        super().__init__(*args, **kwargs)
