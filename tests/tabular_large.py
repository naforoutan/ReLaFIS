"""
Large-scale tabular datasets (tens of thousands of rows).

These loaders share a common pattern: the data file is large enough that the
requested train_size can exceed the actual dataset size, so each class has a
guard that falls back to 80% of available rows when that happens.

To add a new large dataset here:
  1. Load and clean the data.
  2. Apply the same train_size guard pattern used below.
  3. Call super().__init__() as usual.
"""

import pandas as pd
from .base import Test


class AdultIncome(Test):
    """UCI Adult Income dataset (~48 000 rows, binary ≤50K / >50K)."""

    _COLUMNS = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income",
    ]
    _DEFAULT_TRAIN = 26048

    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv(
            "./data/adult.data",
            header=None,
            names=self._COLUMNS,
            skipinitialspace=True,
        )
        self.target = df["income"].map({"<=50K": 0, ">50K": 1}).dropna()
        self.df = df.loc[self.target.index].drop("income", axis=1)
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "income"

        actual_n = len(self.df)
        train_size = kwargs.get("train_size", self._DEFAULT_TRAIN)
        if train_size > actual_n:
            train_size = int(actual_n * 0.8)
            print(f"AdultIncome: requested train_size exceeds {actual_n}; using {train_size}.")
        kwargs["train_size"] = train_size

        super().__init__(*args, **kwargs)


class BankMarketing(Test):
    """Bank Marketing dataset (~45 000 rows, binary subscription target)."""

    _DEFAULT_TRAIN = 36168

    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("./data/bank-full.csv", sep=",", header=0)
        self.target = df["Target"].map({"yes": 1, "no": 0}).dropna()
        self.df = df.drop("Target", axis=1).loc[self.target.index]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "Target"

        actual_n = len(self.df)
        train_size = kwargs.get("train_size", self._DEFAULT_TRAIN)
        if train_size > actual_n:
            train_size = int(actual_n * 0.8)
            print(f"BankMarketing: requested train_size exceeds {actual_n}; using {train_size}.")
        kwargs["train_size"] = train_size

        super().__init__(*args, **kwargs)


class Smoke(Test):
    """Smoke detection IoT sensor dataset (~62 000 rows, binary fire alarm)."""

    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("data/smoke/smoke_detection_iot.csv")
        df.drop(["index", "UTC"], axis=1, inplace=True)
        self.df = df.drop("Fire Alarm", axis=1)
        self.target = df["Fire Alarm"]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "Fire Alarm"
        print(f"Smoke: {self.df.shape}")

        if "train_size" not in kwargs:
            kwargs["train_size"] = int(len(self.df) * 0.8)

        super().__init__(*args, **kwargs)
