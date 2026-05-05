import os

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


class MFeat(Test):
    def __init__(self, *args, **kwargs) -> None:
        import os
        
        # Define file paths (files are in ./data/mfeat/ directory)
        features = {
            'factors': './data/mfeat/mfeat-fac',
            'fourier': './data/mfeat/mfeat-fou',
            'karhunen': './data/mfeat/mfeat-kar',
            'morph': './data/mfeat/mfeat-mor',
            'pixel': './data/mfeat/mfeat-pix',
            'zernike': './data/mfeat/mfeat-zer'
        }
        
        # Read all features and combine with unique column names
        data_frames = []
        for name, path in features.items():
            if os.path.exists(path):
                df = pd.read_csv(path, sep='\s+', header=None)
                # Rename columns to include feature type prefix
                df.columns = [f"{name}_{i}" for i in range(df.shape[1])]
                data_frames.append(df)
                print(f"Loaded {name} with shape {df.shape}")
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        # Combine all features horizontally
        self.df = pd.concat(data_frames, axis=1)
        
        # Generate correct labels (200 samples per digit 0-9)
        n_samples_total = len(self.df)
        samples_per_digit = 200
        
        # Create labels: 0 (200x), 1 (200x), ..., 9 (200x)
        labels = [digit for digit in range(10) for _ in range(samples_per_digit)]
        
        # Create target series
        self.target = pd.Series(labels[:n_samples_total], name='digit')
        
        print(f"Created labels for {len(self.target)} samples")
        print(f"Class distribution:\n{self.target.value_counts().sort_index()}")
        print(f"Total features: {self.df.shape[1]}")
        
        # No need to rename columns again as string - they're already strings with prefixes
        self.target.name = 'digit'
        
        # MFeat has 2000 samples total, 200 per digit (0-9)
        super().__init__(train_size=1600, *args, **kwargs)


class Autism(Test):
    def __init__(self, *args, **kwargs) -> None:
        from scipy.io import arff
        import numpy as np
        
        path = "./data/Autism.arff"
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        
        # Decode bytes to string if needed
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode('utf-8')
        
        # Target is last column 'Class/ASD'
        self.target = df['Class/ASD'].map({'YES': 1, 'NO': 0})
        self.df = df.drop('Class/ASD', axis=1)
        
        # Remove identifier columns
        if 'age_desc' in self.df.columns:
            self.df = self.df.drop('age_desc', axis=1)
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = 'ASD'
        
        super().__init__(train_size=560, *args, **kwargs)


class Digits(Test):
    def __init__(self, *args, **kwargs) -> None:
        # Built-in dataset from sklearn
        data = datasets.load_digits(as_frame=True)
        self.target = data.target
        self.df = data.data
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = 'digit'
        
        super().__init__(train_size=1437, *args, **kwargs)


class DNA(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/promoters.data"
        
        # Read the DNA promoter dataset
        with open(path, 'r') as f:
            lines = f.readlines()
        
        data = []
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.strip().split(',')
                sequence_class = parts[0]  # '+' or '-'
                sequence = ''.join(parts[1:]).replace('"', '')
                data.append([sequence_class, sequence])
        
        df = pd.DataFrame(data, columns=['class', 'sequence'])
        
        # Convert DNA sequence to numerical features (one-hot encode nucleotides)
        nucleotides = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        max_len = max(df['sequence'].str.len())
        
        for i in range(max_len):
            df[f'pos_{i}'] = df['sequence'].apply(
                lambda x: nucleotides.get(x[i] if i < len(x) else 'A', 0)
            )
        
        self.target = df['class'].map({'+': 1, '-': 0})
        self.df = df.drop(['class', 'sequence'], axis=1)
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = 'promoter'
        
        super().__init__(train_size=80, *args, **kwargs)


class SyntheticGaussian(Test):
    def __init__(self, n_clusters=5, n_samples=1000, *args, **kwargs) -> None:
        # Generate synthetic 2D Gaussian clusters
        import numpy as np
        
        np.random.seed(42)
        
        # Create cluster centers
        centers = np.random.randn(n_clusters, 2) * 3
        
        # Generate samples around centers
        samples_per_cluster = n_samples // n_clusters
        X = []
        y = []
        
        for i, center in enumerate(centers):
            cluster_samples = np.random.randn(samples_per_cluster, 2) * 0.5 + center
            X.extend(cluster_samples)
            y.extend([i] * samples_per_cluster)
        
        # Add remaining samples
        remaining = n_samples - len(X)
        if remaining > 0:
            last_center = centers[-1]
            extra_samples = np.random.randn(remaining, 2) * 0.5 + last_center
            X.extend(extra_samples)
            y.extend([n_clusters-1] * remaining)
        
        X = np.array(X)
        y = np.array(y)
        
        self.df = pd.DataFrame(X, columns=['x', 'y'])
        self.target = pd.Series(y, name='cluster')
        
        # Shuffle
        indices = np.random.permutation(len(self.df))
        self.df = self.df.iloc[indices].reset_index(drop=True)
        self.target = self.target.iloc[indices].reset_index(drop=True)
        
        self.df.columns = self.df.columns.astype(str)
        self.target.name = 'cluster'
        
        train_size = kwargs.pop('train_size', int(n_samples * 0.8))
        super().__init__(train_size=train_size, *args, **kwargs)