from .base import Test

from .tabular_small import (
    Iris, Wine, Glass, Heart, Haberman, Thyroid,
    Cryotheraphy, Immunotherapy, BreastCancer, PimaDiabetes, PimaDiabetesCase2,
    BCW, Autism, CarEvaluation, SegmentaitionUCI,
    DNA, Digits, DigitsUCI as Digits_UCI, Parkinson,
    MFeat, SyntheticGaussian,
)
from .tabular_large import AdultIncome, BankMarketing, Smoke
from .image import MNIST, FashionMNIST
from .bio_medical import Diabetes, Digits_UCI_Repo, Isolet

__all__ = [
    "Test",
    "Iris", "Wine", "Glass", "Heart", "Haberman", "Thyroid",
    "Cryotheraphy", "Immunotherapy", "BreastCancer", "PimaDiabetes", "PimaDiabetesCase2",
    "BCW", "Autism", "CarEvaluation", "SegmentaitionUCI",
    "DNA", "Digits", "Digits_UCI", "Parkinson", "MFeat", "SyntheticGaussian",
    "AdultIncome", "BankMarketing", "Smoke",
    "MNIST", "FashionMNIST", "ORL",
    "Diabetes", "Digits_UCI_Repo", "Isolet",
]
