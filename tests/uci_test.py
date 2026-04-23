import pandas as pd
from sklearn import datasets
from . import Test
from ucimlrepo import fetch_ucirepo


class Cryotheraphy(Test):
    def __init__(self, *args, **kwargs) -> None:
        file_path = './data/Cryotherapy.xlsx'
        self.df = pd.read_excel(file_path)
        self.target = 'Result_of_Treatment'
        super().__init__(train_size=63, *args, **kwargs)


class Haberman(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/haberman.data"
        df = pd.read_csv(path, header=None)
        
        self.target = df[3]           # This is a pandas Series
        self.df = df.drop(columns=[3])  # Remove target from features
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name)
        
        super().__init__(train_size=214, *args, **kwargs)


class Heart(Test):
    def __init__(self, *args, **kwargs) -> None:
        # Path to the local Cleveland dataset
        path = "./data/heart.data"
        df = pd.read_csv(path, header=None, na_values='?')
        df = df.dropna(subset=[df.columns[-1]])
        
        self.df = df.iloc[:, :-1]   # all columns except last
        self.target = df.iloc[:, -1]  # last column
        self.target = (self.target > 0).astype(int)

        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name)
        
        super().__init__(train_size=189, *args, **kwargs)


class Glass(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/glass.data"
        df = pd.read_csv(path, header=None)

        self.df = df.iloc[:, :-1]
        self.target = df.iloc[:, -1]
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name)
        
        super().__init__(train_size=160, *args, **kwargs)


class Segmentaition(Test):
    def __init__(self, *args, **kwargs) -> None:
        file_path = 'data/segmentation.data'
        
        data = pd.read_csv(
            file_path,
            sep=',',
            header=None,
            comment=';',          # skip comment lines starting with ';'
            skip_blank_lines=True,
            engine='python'
        )
        
        self.target = data[0]
        self.df = data.iloc[:, 1:]
        
        before = len(self.df)
        self.df = self.df.dropna()
        self.target = self.target.loc[self.df.index]
        after = len(self.df)
        if after < before:
            print(f"Dropped {before - after} rows with missing values.")
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = 'class'
        
        actual_size = len(self.df)
        train_size = int(actual_size * 0.8)   # 80% of data
        kwargs['train_size'] = train_size
        
        super().__init__(*args, **kwargs)


class Wine(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/wine.data"
        self.df = pd.read_csv(path, header=None)
        self.target = 0   # column index 0 is the target

        self.df.columns = self.df.columns.astype(str)
        # Also ensure target Series has a string name:
        if isinstance(self.target, pd.Series):
            self.target.name = str(self.target.name)
        super().__init__(train_size=124, *args, **kwargs)


class Thyroid(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/thyroid.data"
        self.df = pd.read_csv(path, header=None)
        self.target = 0
        super().__init__(train_size=150, *args, **kwargs)


class Immunotherapy(Test):
    def __init__(self, *args, **kwargs) -> None:

        file_path = './data/Immunotherapy.xlsx'
        data = pd.read_excel(file_path)

        self.df = data.drop('Result_of_Treatment', axis=1)
        self.target = data['Result_of_Treatment']

        super().__init__(train_size=63  , *args, **kwargs)


class Iris(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = datasets.load_iris(as_frame=True)
        self.target = data.target
        self.df = data.data
        super().__init__(train_size=105, *args, **kwargs)



class BreastCancer(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/wdbc.data"
        df = pd.read_csv(path, header=None)

        self.target = df[1].map({'M': 1, 'B': 0})
        self.df = df.iloc[:, 2:]
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name) if self.target.name is not None else 'target'
        
        super().__init__(train_size=455, *args, **kwargs)


class AdultIncome(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/adult.data"
        column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                        'marital-status', 'occupation', 'relationship', 'race', 'sex',
                        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                        'income']
        df = pd.read_csv(path, header=None, names=column_names, skipinitialspace=True)
        
        self.target = df['income'].map({'<=50K': 0, '>50K': 1})
        
        self.target = self.target.dropna()
        self.df = df.loc[self.target.index].drop('income', axis=1)
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = 'income'
        
        actual_n = len(self.df)
        train_size = kwargs.get('train_size', 26048)
        if train_size > actual_n:
            train_size = int(actual_n * 0.8)   # fallback to 80% of data
            print(f"Note: requested train_size=26048 exceeds data size {actual_n}; using {train_size} instead.")
        kwargs['train_size'] = train_size
        
        super().__init__(*args, **kwargs)


class BankMarketing(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/bank-full.csv"
        df = pd.read_csv(path, sep=',', header=0)
        
        self.target = df['Target'].map({'yes': 1, 'no': 0})
        self.df = df.drop('Target', axis=1)
        
        self.target = self.target.dropna()
        self.df = self.df.loc[self.target.index]
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = 'Target'
        
        actual_n = len(self.df)
        train_size = kwargs.get('train_size', 36168)
        if train_size > actual_n:
            train_size = int(actual_n * 0.8)
            print(f"Note: requested train_size=36168 exceeds data size {actual_n}; using {train_size} instead.")
        kwargs['train_size'] = train_size
        
        super().__init__(*args, **kwargs)



class PimaDiabetes(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/diabetes.data"
        df = pd.read_csv(path, header=None)

        self.target = df.iloc[:, -1]      # pandas Series
        self.df = df.iloc[:, :-1]         # features
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = str(self.target.name) if self.target.name is not None else 'target'
        
        super().__init__(train_size=614, *args, **kwargs)


class CarEvaluation(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/car.data"
        column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        df = pd.read_csv(path, header=None, names=column_names)
        # Target: class (unacc, acc, good, vgood) -> we can map to numeric
        # For binary classification? The original datasets include multiclass;
        # you may keep as is or binarize. Here we keep as categorical codes.
        # If you need binary: map 'unacc'/'acc' vs 'good'/'vgood'? Unclear.
        # I'll assume you want multiclass (as in Glass, Wine, etc.)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        self.target = le.fit_transform(df['class'])
        self.df = df.drop('class', axis=1)
        # Optionally one‑hot encode categorical features? Leave as is (strings) –
        # parent code may need to handle. Add note.
        super().__init__(train_size=1384, *args, **kwargs)