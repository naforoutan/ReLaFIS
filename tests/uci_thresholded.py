"""
Real-world UCI benchmark datasets (binarised for classification).

GIFTSHIFT vs LitANFIS benchmark — these are originally regression datasets
from the UCI Machine Learning Repository, fetched live via `ucimlrepo`.
Regression targets are binarised using the thresholds documented in the
benchmark reference (see each class docstring) to create classification
tasks, so they plug into the same `Test` base class used everywhere else
in this project.

Live fetch
----------
All classes here pull data from the UCI API at runtime via
`ucimlrepo.fetch_ucirepo(id=...)`. This requires an internet connection
and the `ucimlrepo` package (`pip install ucimlrepo`).

  - EnergyEfficiencyHeating : Heating_Load > 20 kWh         (high demand)
  - EnergyEfficiencyCooling : Cooling_Load <= 25 kWh        (cool zone)
  - Airfoil                : Sound_Pressure_Level > 125 dB (nonlinear sharp)
  - Concrete                : Strength > 40 MPa             (binary, smooth Gaussian)
  - ConcreteMulticlass      : low / medium / high MPa       (3-class)
  - CCPP                    : Electrical_Output > 480 MW    (sharp linear)

To add a new dataset here:
  - Use ucimlrepo.fetch_ucirepo(id=<id>) for live fetch.
  - Binarise (or bucket) the continuous target with the documented threshold.
  - Set self.df = features and self.target = the binarised label, then call
    super().__init__() as usual.
"""

import pandas as pd
from .base import Test


class EnergyEfficiencyHeating(Test):
    """Dataset 6 - Energy Efficiency (heating load variant).

    Building energy simulation data (Tsanas & Xifara, 2012). Features
    describe building geometry and glazing. The orientation feature
    creates a sharp threshold; compactness creates a smooth Gaussian-like
    boundary — ideal for testing comb_weight adaptation.

    Inequality: Heating_Load > 20 kWh (high demand)
    8 geometric features, 768 samples, mixed sharp/smooth boundary.
    UCI id=242.
    """

    def __init__(self, *args, threshold=20.0, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=242)
        X = data.data.features
        y = data.data.targets

        # Targets dataframe has both heating and cooling load columns;
        # heating load is conventionally the first ("Y1").
        heating_col = y.columns[0]

        self.df = X
        self.target = (y[heating_col] > threshold).astype(int)
        self.target.name = "label"

        print(f"✅ EnergyEfficiencyHeating loaded: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, threshold={threshold}, "
              f"label balance {dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class EnergyEfficiencyCooling(Test):
    """Dataset 6 (alt) - Energy Efficiency (cooling load variant).

    Same underlying data as EnergyEfficiencyHeating, but using the
    alternative inequality on the cooling load target instead.

    Inequality: Cooling_Load <= 25 kWh (cool zone)
    8 geometric features, 768 samples, mixed sharp/smooth boundary.
    UCI id=242.
    """

    def __init__(self, *args, threshold=25.0, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=242)
        X = data.data.features
        y = data.data.targets

        # Cooling load is conventionally the second target column ("Y2").
        cooling_col = y.columns[1] if y.shape[1] > 1 else y.columns[0]

        self.df = X
        self.target = (y[cooling_col] <= threshold).astype(int)
        self.target.name = "label"

        print(f"✅ EnergyEfficiencyCooling loaded: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, threshold={threshold}, "
              f"label balance {dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class Airfoil(Test):
    """Dataset 7 - Airfoil Self-Noise.

    NASA wind-tunnel aeroacoustic measurements. The boundary between
    quiet and loud operation is highly nonlinear due to turbulence
    interactions. Smallest real dataset (1 503 samples), so data
    efficiency is also being tested alongside accuracy.

    Inequality: Sound_Pressure_Level > 125 dB
    Features: frequency (Hz), angle of attack (deg), chord length (m),
              free-stream velocity (m/s), suction-side displacement
              thickness (m).
    5 features, 1 503 samples, nonlinear sharp boundary.
    UCI id=291.
    """

    def __init__(self, *args, threshold=125.0, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=291)
        X = data.data.features
        y = data.data.targets
        target_col = y.columns[0]

        self.df = X
        self.target = (y[target_col] > threshold).astype(int)
        self.target.name = "label"

        print(f"✅ Airfoil loaded: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, threshold={threshold} dB, "
              f"label balance {dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class Concrete(Test):
    """Dataset 8 - Concrete Compressive Strength (binary).

    Concrete curing dataset (Yeh, 1998). The age feature has a strong
    nonlinear effect following a Gaussian-like growth curve, making this
    the dataset most likely to favour LitANFIS's pure Gaussian membership
    functions.

    Inequality: Strength > 40 MPa (high-performance)
    Features: cement, blast furnace slag, fly ash, water, superplasticizer,
              coarse aggregate, fine aggregate, age.
    8 features, 1 030 samples, smooth Gaussian boundary.
    UCI id=165.
    """

    def __init__(self, *args, threshold=40.0, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=165)
        X = data.data.features
        y = data.data.targets
        target_col = y.columns[0]

        self.df = X
        self.target = (y[target_col] > threshold).astype(int)
        self.target.name = "label"

        print(f"✅ Concrete loaded: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, threshold={threshold} MPa, "
              f"label balance {dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class ConcreteMulticlass(Test):
    """Dataset 8 (alt) - Concrete Compressive Strength (3-class).

    Same underlying data as Concrete, bucketed into three classes instead
    of a binary split, per the benchmark reference's multiclass option.
    Adds multiclass complexity on top of the curved age-strength
    relationship.

    Inequality: Strength <= 25 MPa -> low
                25 < Strength <= 40 MPa -> medium
                Strength > 40 MPa -> high
    8 features, 1 030 samples, smooth Gaussian boundary, 3 classes.
    UCI id=165.
    """

    def __init__(self, *args, low_threshold=25.0, high_threshold=40.0, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=165)
        X = data.data.features
        y = data.data.targets
        target_col = y.columns[0]

        strength = y[target_col]
        label = pd.cut(
            strength,
            bins=[-float("inf"), low_threshold, high_threshold, float("inf")],
            labels=["low", "medium", "high"],
        )

        self.df = X
        self.target = label
        self.target.name = "label"

        print(f"✅ ConcreteMulticlass loaded: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, thresholds=({low_threshold}, "
              f"{high_threshold}) MPa, class distribution "
              f"{dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


class CCPP(Test):
    """Dataset 9 - Combined Cycle Power Plant.

    Ambient sensor readings from a gas turbine plant over six years.
    Relationships are nearly linear, and temperature thresholds for
    turbine mode switches create crisp boundaries — strongly favouring
    sigmoid membership. Largest dataset (9 568 samples) tests scalability.

    Inequality: Electrical_Output > 480 MW
    Features: temperature (°C), exhaust vacuum (cm Hg), ambient pressure
              (mbar), relative humidity (%).
    4 features, 9 568 samples, sharp linear boundary.
    UCI id=294.
    """

    def __init__(self, *args, threshold=480.0, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=294)
        X = data.data.features
        y = data.data.targets
        target_col = y.columns[0]

        self.df = X
        self.target = (y[target_col] > threshold).astype(int)
        self.target.name = "label"

        print(f"✅ CCPP loaded: {self.df.shape[0]} samples, "
              f"{self.df.shape[1]} features, threshold={threshold} MW, "
              f"label balance {dict(self.target.value_counts())}")

        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Future slots — add below as you test new UCI regression-turned-classification papers
# ---------------------------------------------------------------------------
# class WineQuality(Test):    ...
# class HousingPrices(Test):  ...