"""
Small and medium classic tabular classification datasets.

Covers well-known benchmarks typically used in fuzzy/neuro-fuzzy papers:
UCI repository classics, sklearn built-ins, and a few local-file loaders.

To add a new dataset here:
  1. Subclass Test (imported from tests/__init__.py via relative import)
  2. Set self.df (features DataFrame) and self.target (Series or column index)
  3. Call super().__init__(...) with an appropriate train_size
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
        super().__init__(train_size=105, *args, **kwargs)


class Digits(Test):
    """Sklearn built-in 8×8 digit images (1 797 samples, 64 features)."""
    def __init__(self, *args, **kwargs) -> None:
        data = datasets.load_digits(as_frame=True)
        self.target = data.target
        self.df = data.data
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "digit"
        super().__init__(train_size=1437, *args, **kwargs)


# ---------------------------------------------------------------------------
# Local-file loaders — medical / treatment
# ---------------------------------------------------------------------------

class Cryotheraphy(Test):
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_excel("./data/Cryotherapy.xlsx")
        self.target = "Result_of_Treatment"
        super().__init__(train_size=63, *args, **kwargs)


class Immunotherapy(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = pd.read_excel("./data/Immunotherapy.xlsx")
        self.df = data.drop("Result_of_Treatment", axis=1)
        self.target = data["Result_of_Treatment"]
        super().__init__(train_size=63, *args, **kwargs)


class PimaDiabetes(Test):
    """Pima Indians Diabetes dataset (local .data file)."""
    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("./data/diabetes.data", header=None)
        self.target = df.iloc[:, -1]
        self.df = df.iloc[:, :-1]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name) if self.target.name is not None else "target"
        super().__init__(train_size=614, *args, **kwargs)


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
        super().__init__(train_size=455, *args, **kwargs)


class Autism(Test):
    def __init__(self, *args, **kwargs) -> None:
        from scipy.io import arff

        data, _ = arff.loadarff("./data/Autism.arff")
        df = pd.DataFrame(data)
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode("utf-8")

        self.target = df["Class/ASD"].map({"YES": 1, "NO": 0})
        self.df = df.drop("Class/ASD", axis=1)
        if "age_desc" in self.df.columns:
            self.df = self.df.drop("age_desc", axis=1)
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "ASD"
        super().__init__(train_size=560, *args, **kwargs)


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
        super().__init__(train_size=160, *args, **kwargs)


class Wine(Test):
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("./data/wine.data", header=None)
        self.target = 0
        self.df.columns = self.df.columns.astype(str)
        super().__init__(train_size=124, *args, **kwargs)


class Thyroid(Test):
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("./data/thyroid.data", header=None)
        self.target = 0
        super().__init__(train_size=150, *args, **kwargs)


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

        if "train_size" not in kwargs:
            kwargs["train_size"] = int(len(self.df) * 0.8)
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
        super().__init__(train_size=189, *args, **kwargs)


class Haberman(Test):
    """Haberman's survival dataset."""
    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("./data/haberman.data", header=None)
        self.target = df[3]
        self.df = df.drop(columns=[3])
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name)
        super().__init__(train_size=214, *args, **kwargs)


# ---------------------------------------------------------------------------
# Local-file loaders — image / digit (tabular representation)
# ---------------------------------------------------------------------------

class DigitsUCI(Test):
    """Optical digits from local .tra/.tes split files (tabular pixel features)."""
    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("./data/unfis_data/optdigits.tra", header=None)
        test_data = pd.read_csv("./data/unfis_data/optdigits.tes", header=None)
        self.target = 64
        super().__init__(*args, **kwargs, test_data=test_data, index=False, train_size=3823)


# ---------------------------------------------------------------------------
# Local-file loaders — text / sequence
# ---------------------------------------------------------------------------

class DNA(Test):
    """Promoter DNA sequences, one-hot encoded per nucleotide position."""
    def __init__(self, *args, **kwargs) -> None:
        with open("./data/promoters.data") as f:
            lines = f.readlines()

        nucleotides = {"A": 0, "C": 1, "G": 2, "T": 3}
        data = []
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split(",")
                seq_class = parts[0]
                sequence = "".join(parts[1:]).replace('"', "")
                data.append([seq_class, sequence])

        df = pd.DataFrame(data, columns=["class", "sequence"])
        max_len = df["sequence"].str.len().max()
        for i in range(max_len):
            df[f"pos_{i}"] = df["sequence"].apply(
                lambda x: nucleotides.get(x[i] if i < len(x) else "A", 0)
            )

        self.target = df["class"].map({"+": 1, "-": 0})
        self.df = df.drop(["class", "sequence"], axis=1)
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "promoter"
        super().__init__(train_size=80, *args, **kwargs)


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
        super().__init__(train_size=1600, *args, **kwargs)


# ---------------------------------------------------------------------------
# Local-file loaders — segmentation (two versions)
# ---------------------------------------------------------------------------

class SegmentaitionUCI(Test):
    """Image segmentation dataset loaded from a single local .data file."""
    def __init__(self, *args, **kwargs) -> None:
        data = pd.read_csv(
            "data/segmentation.data",
            sep=",", header=None, comment=";",
            skip_blank_lines=True, engine="python",
        )
        self.target = data[0]
        self.df = data.iloc[:, 1:]

        before = len(self.df)
        self.df = self.df.dropna()
        self.target = self.target.loc[self.df.index]
        if len(self.df) < before:
            print(f"Segmentation: dropped {before - len(self.df)} rows with NaNs.")

        self.df.columns = self.df.columns.astype(str)
        self.target.name = "class"
        kwargs["train_size"] = kwargs.get("train_size", int(len(self.df) * 0.8))
        super().__init__(*args, **kwargs)


class Segmentation(Test):
    """Image segmentation dataset built from two split files (.data + .test)."""
    def __init__(self, *args, **kwargs) -> None:
        df1 = pd.read_csv("./data/unfis_data/segmentation.data", header=None)
        df2 = pd.read_csv("./data/unfis_data/segmentation.test", header=None)
        self.df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
        self.df = self.df.drop(1, axis=1)
        self.target = 0
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
        super().__init__(train_size=1384, *args, **kwargs)


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

        train_size = kwargs.pop("train_size", int(n_samples * 0.8))
        super().__init__(train_size=train_size, *args, **kwargs)
