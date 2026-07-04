"""
Large-scale tabular datasets (tens of thousands of rows).

All loaders inherit the 70/30 train/test split from tests.base.Test.
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

        super().__init__(*args, **kwargs)


class BankMarketing(Test):
    """Bank Marketing dataset (~45 000 rows, binary subscription target)."""

    def __init__(self, *args, **kwargs) -> None:
        df = pd.read_csv("./data/bank-full.csv", sep=",", header=0)
        self.target = df["Target"].map({"yes": 1, "no": 0}).dropna()
        self.df = df.drop("Target", axis=1).loc[self.target.index]
        self.df.columns = self.df.columns.astype(str)
        self.target.name = "Target"

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

        super().__init__(*args, **kwargs)
