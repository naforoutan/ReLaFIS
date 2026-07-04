"""
Small and medium classic tabular classification datasets.

Covers well-known benchmarks typically used in fuzzy/neuro-fuzzy papers:
UCI repository classics, sklearn built-ins, and a few local-file loaders.

To add a new dataset here:
  1. Subclass Test (imported from tests/__init__.py via relative import)
  2. Set self.df (features DataFrame) and self.target (Series or column index)
  3. Call super().__init__(...) — train/test split is 70/30 by default (see base.Test)
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from .base import Test


# ---------------------------------------------------------------------------
# sklearn built-ins
# ---------------------------------------------------------------------------

class Iris(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = datasets.load_iris(as_frame=True)
        self.target = data.target
        self.df = data.data
        super().__init__(*args, **kwargs)


class Digits(Test):
    """Sklearn built-in 8×8 digit images (1 797 samples, 64 features)."""
    def __init__(self, *args, **kwargs) -> None:
        data = datasets.load_digits(as_frame=True)
        self.target = data.target
        self.df = data.data
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "digit"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Local-file loaders — medical / treatment
# ---------------------------------------------------------------------------

class Cryotheraphy(Test):
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_excel("./data/Cryotherapy.xlsx")
        self.target = "Result_of_Treatment"
        super().__init__(*args, **kwargs)


class Immunotherapy(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = pd.read_excel("./data/Immunotherapy.xlsx")
        self.df = data.drop("Result_of_Treatment", axis=1)
        self.target = data["Result_of_Treatment"]
        super().__init__(*args, **kwargs)


def _fetch_pima_online():
    """Fetch the UCI Pima Indians Diabetes dataset from OpenML (id=37).

    The classic Pima dataset was removed from the UCI API, but OpenML mirrors
    the identical 768-sample / 8-feature version. Returns ``(features_df,
    target_series)`` with the target mapped to 1 (positive) / 0 (negative).

    Requires an internet connection and scikit-learn.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml("diabetes", version=1, as_frame=True)
    features = data.data.copy()
    target = data.target.astype(str).map(
        {"tested_positive": 1, "tested_negative": 0}
    )
    features.columns = features.columns.astype(str)
    target.name = "class"
    return features, target


class PimaDiabetes(Test):
    """Pima Indians Diabetes dataset — Case 1 (fetched online from OpenML/UCI).

    Full Pima set: all 768 samples, 8 features, binary target. Medically
    impossible zero values (e.g. glucose/BMI = 0) are kept as-is.

    Requires an internet connection and scikit-learn.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.df, self.target = _fetch_pima_online()
        super().__init__(*args, **kwargs)


class PimaDiabetesCase2(Test):
    """Pima Indians Diabetes dataset — Case 2 (fetched online from OpenML/UCI).

    Rows containing medically impossible zeros in any of plasma glucose,
    diastolic blood pressure, triceps skin-fold thickness, 2-hour serum
    insulin, or BMI are dropped, leaving ~392 complete samples.

    Requires an internet connection and scikit-learn.
    """
    # Feature columns where 0 is physiologically impossible and therefore
    # treated as a missing value that disqualifies the row.
    _IMPOSSIBLE_ZERO_COLS = ["plas", "pres", "skin", "insu", "mass"]

    def __init__(self, *args, **kwargs) -> None:
        df, target = _fetch_pima_online()

        before = len(df)
        mask = (df[self._IMPOSSIBLE_ZERO_COLS] == 0).any(axis=1)
        self.df = df[~mask].reset_index(drop=True)
        self.target = target[~mask].reset_index(drop=True)
        if len(self.df) < before:
            print(f"PimaDiabetesCase2: dropped {before - len(self.df)} rows with impossible zeros "
                  f"({len(self.df)} samples remain).")

        super().__init__(*args, **kwargs)


class BCW(Test):
    """Breast Cancer Wisconsin (original, local .data file)."""
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("./data/unfis_data/breast-cancer-wisconsin.data", header=None)
        self.df.drop(0, axis=1, inplace=True)
        self.target = 9
        print(self.df.shape)
        super().__init__(*args, **kwargs)


class BreastCancer(Test):
    """WDBC breast-cancer dataset (30 features, binary M/B label)."""
    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("./data/wdbc.data", header=None)
        self.target = df[1].map({"M": 1, "B": 0})
        self.df = df.iloc[:, 2:]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name) if self.target.name is not None else "target"
        super().__init__(*args, **kwargs)


class Autism(Test):
    """Autism screening dataset (704 samples, binary ASD vs non-ASD).

    Drops ``result`` (aggregate screening score — perfect proxy for the label)
    and ``age_desc`` (redundant with ``age``). Uses simple imputation because
    only ``age`` has missing values (2 rows).
    """
    def __init__(self, *args, **kwargs) -> None:
        from scipy.io import arff

        data, _ = arff.loadarff("./data/Autism.arff")
        df = pd.DataFrame(data)
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode("utf-8")

        self.target = df["Class/ASD"].map({"YES": 1, "NO": 0})
        drop_cols = ["Class/ASD", "result"]
        if "age_desc" in df.columns:
            drop_cols.append("age_desc")
        self.df = df.drop(columns=drop_cols)
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "ASD"

        kwargs.setdefault("imputation_type", "simple")
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Local-file loaders — physical / natural science
# ---------------------------------------------------------------------------

class Glass(Test):
    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("./data/glass.data", header=None)
        self.df = df.iloc[:, :-1]
        self.target = df.iloc[:, -1]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name)
        super().__init__(*args, **kwargs)


class Wine(Test):
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("./data/wine.data", header=None)
        self.target = 0
        self.df.columns = self.df.columns.astype(str)
        super().__init__(*args, **kwargs)


class Thyroid(Test):
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("./data/thyroid.data", header=None)
        self.target = 0
        super().__init__(*args, **kwargs)


class Parkinson(Test):
    def __init__(self, *args, **kwargs) -> None:
        from scipy.io import loadmat

        mat = loadmat("./data/parkinson.mat")
        feature_keys = [k for k in mat if not k.startswith("__") and k not in ("sample_source", "label")]
        self.df = pd.DataFrame({k: mat[k].flatten() for k in feature_keys})
        self.target = pd.Series(mat["label"].flatten(), name="label").astype(int)
        self.df.columns = self.df.columns.astype(str)

        print(f"Parkinson: {len(self.df)} samples, {self.df.shape[1]} features")
        print(f"Class distribution:\n{self.target.value_counts().sort_index()}")

        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Local-file loaders — cardiovascular / survival
# ---------------------------------------------------------------------------

class Heart(Test):
    """Cleveland heart disease dataset (binary: disease present/absent)."""
    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("./data/heart.data", header=None, na_values="?")
        df = df.dropna(subset=[df.columns[-1]])
        self.df = df.iloc[:, :-1]
        self.target = (df.iloc[:, -1] > 0).astype(int)
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name)
        super().__init__(*args, **kwargs)


'''class Haberman(Test):
    """Haberman's survival dataset."""
    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("./data/haberman.data", header=None)
        self.target = df[3]
        self.df = df.drop(columns=[3])
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name)
        super().__init__(train_size=214, *args, **kwargs)'''

class Haberman(Test):
    def __init__(self, *args, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=43)

        self.df = data.data.features
        self.target = data.data.targets

        target_column_name = self.target.columns[0]
        data = pd.concat([self.df, self.target], axis=1)
        data = data.dropna(subset=[target_column_name])

        self.df = data.iloc[:, :-1]
        self.target = data.iloc[:, -1]
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Local-file loaders — image / digit (tabular representation)
# ---------------------------------------------------------------------------

class DigitsUCI(Test):
    """Optical digits from local .tra/.tes files (tabular pixel features)."""
    def __init__(self, *args, **kwargs) -> None:
        train_df = pd.read_csv("./data/unfis_data/optdigits.tra", header=None)
        test_df = pd.read_csv("./data/unfis_data/optdigits.tes", header=None)
        self.df = pd.concat([train_df, test_df], ignore_index=True)
        self.target = 64
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Online loaders — text / sequence
# ---------------------------------------------------------------------------

class DNA(Test):
    """StatLog DNA dataset — splice-junction gene sequences (fetched online).

    Fetched from OpenML ('dna', id=40670): 3186 samples, 180 binary features
    (60 nucleotide positions one-hot encoded into 3 indicators each) and a
    3-class target (1, 2, 3) for the splice-junction type.

    Requires an internet connection and scikit-learn.
    """
    def __init__(self, *args, **kwargs) -> None:
        from sklearn.datasets import fetch_openml

        data = fetch_openml("dna", version=1, as_frame=True)
        self.df = data.data.apply(pd.to_numeric).astype(int)
        self.df.columns = self.df.columns.astype(str)
        self.target = data.target.astype(int)
        self.target.name = "class"
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Local-file loaders — multi-view / high-dimensional
# ---------------------------------------------------------------------------

class MFeat(Test):
    """Multiple Features dataset: six feature sets for handwritten digit recognition."""
    def __init__(self, *args, **kwargs) -> None:
        import os

        feature_files = {
            "factors":  "./data/mfeat/mfeat-fac",
            "fourier":  "./data/mfeat/mfeat-fou",
            "karhunen": "./data/mfeat/mfeat-kar",
            "morph":    "./data/mfeat/mfeat-mor",
            "pixel":    "./data/mfeat/mfeat-pix",
            "zernike":  "./data/mfeat/mfeat-zer",
        }
        frames = []
        for name, path in feature_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"MFeat file not found: {path}")
            df = pd.read_csv(path, sep=r"\s+", header=None)
            df.columns = [f"{name}_{i}" for i in range(df.shape[1])]
            frames.append(df)
            print(f"MFeat loaded {name}: {df.shape}")

        self.df = pd.concat(frames, axis=1)
        n = len(self.df)
        labels = [digit for digit in range(10) for _ in range(200)]
        self.target = pd.Series(labels[:n], name="digit")
        print(f"MFeat total: {n} samples, {self.df.shape[1]} features")
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# UCI fetch — image segmentation (tabular region features)
# ---------------------------------------------------------------------------

class SegmentaitionUCI(Test):
    """Image Segmentation dataset fetched live from UCI (id=50).

    Each row is a 3×3 image region with 19 texture/color features and a
    material class label (7 classes). Requires: pip install ucimlrepo
    """
    def __init__(self, *args, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=50)
        self.df = data.data.features
        self.target = data.data.targets

        target_column_name = self.target.columns[0]
        combined = pd.concat([self.df, self.target], axis=1)
        before = len(combined)
        combined = combined.dropna(subset=[target_column_name])
        if len(combined) < before:
            print(f"Segmentation: dropped {before - len(combined)} rows with NaNs.")

        self.df = combined.iloc[:, :-1]
        self.target = combined.iloc[:, -1]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "class"
        super().__init__(*args, **kwargs)




# ---------------------------------------------------------------------------
# Local-file loaders — categorical / ordinal
# ---------------------------------------------------------------------------

class CarEvaluation(Test):
    def __init__(self, *args, **kwargs) -> None:
        from sklearn.preprocessing import LabelEncoder

        cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        df = pd.read_csv("./data/car.data", header=None, names=cols)
        le = LabelEncoder()
        self.target = le.fit_transform(df["class"])
        self.df = df.drop("class", axis=1)
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
