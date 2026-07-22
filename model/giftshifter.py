"""Backward-compatible aliases; prefer ``from model.GIFT import GIFT``."""

from model.GIFT import GIFT, MamdaniGIFT, SklearnGIFTWrapper

GIFTSHIFTER = GIFT
MamdaniGIFTSHIFTER = MamdaniGIFT
SklearnGIFTSHIFTERWrapper = SklearnGIFTWrapper

__all__ = [
    "GIFT",
    "MamdaniGIFT",
    "SklearnGIFTWrapper",
    "GIFTSHIFTER",
    "MamdaniGIFTSHIFTER",
    "SklearnGIFTSHIFTERWrapper",
]
