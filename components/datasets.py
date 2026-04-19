import numpy as np
import pandas as pd
from sklearn.datasets import make_moons, make_circles, make_swiss_roll, load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def make_swiss_roll_dataset(n_samples=1000, noise=0.1):
    """3D swiss roll dataset, regression on unrolled coordinate"""
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise)
    # Optional: normalize t to [0,1] as regression target
    t = (t - t.min()) / (t.max() - t.min())
    return X.astype(np.float32), t.astype(np.float32).reshape(-1,1)

# ---------------------------
# Mackey-Glass time series
# ---------------------------

def generate_mackey_glass(n_samples=2000, tau=17, delta_t=1, beta=0.2, gamma=0.1, n_input=10):
    """
    Generate Mackey-Glass time series for regression.
    Returns X (lagged inputs) and y (next value)
    """
    x = np.zeros(n_samples + n_input)
    x[0] = 1.5  # initial value
    for t in range(1, n_samples + n_input - 1):
        x_tau = x[t - tau] if t - tau >= 0 else 0
        x[t + 1] = x[t] + delta_t * (beta * x_tau / (1 + x_tau**10) - gamma * x[t])
    # Create lagged input
    X, y = [], []
    for t in range(n_input, len(x)-1):
        X.append(x[t-n_input:t])
        y.append(x[t])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1,1)
    return X, y


# ---------------------------
# Binary classification datasets
# ---------------------------

def make_spirals(n_samples=2000, noise=0.2):
    """2D spiral dataset, binary classification"""
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1)*noise
    d1y =  np.sin(n)*n + np.random.rand(n_samples,1)*noise
    X = np.vstack((np.hstack((d1x,d1y)), np.hstack((-d1x,-d1y))))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples)))
    return X.astype(np.float32), y.astype(np.int64)

def make_circles_dataset(n_samples=1000, noise=0.05, factor=0.5):
    """2D circle dataset, binary classification"""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    return X.astype(np.float32), y.astype(np.int64)

def make_moons_dataset(n_samples=1000, noise=0.1):
    """2D moons dataset, binary classification"""
    X, y = make_moons(n_samples=n_samples, noise=noise)
    return X.astype(np.float32), y.astype(np.int64)

def load_haberman_dataset():
    # Columns: age, year, nodes, survival_status
    data = np.loadtxt("data/haberman.csv", delimiter=",")
    X = data[:, :3].astype(np.float32)
    y_raw = data[:, 3].astype(np.int64)
    classes = np.unique(y_raw)
    assert len(classes) == 2, "Haberman dataset must be binary"
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[c] for c in y_raw], dtype=np.int64)
    return X, y



def load_cryotherapy_dataset():
    data = np.loadtxt("data/Cryotherapy.csv", delimiter=",")
    y_raw = data[:, 0].astype(np.int64)
    X = data[:, 1:].astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    classes = np.unique(y_raw)
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[c] for c in y_raw], dtype=np.int64)

    return X, y

def load_heart_dataset():
    df = pd.read_csv("data/Heart.csv")
    y = df["AHD"].map({"No": 0, "Yes": 1}).to_numpy(dtype=np.int64)
    X = df.drop(columns=["AHD"])
    chest_pain_map = {
        "typical": 0,
        "nontypical": 1,
        "nonanginal": 2,
        "asymptomatic": 3,
    }
    thal_map = {
        "normal": 0,
        "fixed": 1,
        "reversable": 2,
    }
    X["ChestPain"] = X["ChestPain"].map(chest_pain_map)
    X["Thal"] = X["Thal"].map(thal_map)
    X = X.replace("NA", np.nan)
    X = X.astype(float)
    X = X.fillna(X.median())
    return X.to_numpy(dtype=np.float32), y


def load_autism_dataset():
    # Read CSV with mixed types
    df = pd.read_csv("data/autism.csv")
    y_raw = df.iloc[:, -1]
    classes = y_raw.unique()
    assert len(classes) == 2, "Autism dataset must be binary"
    mapping = {c: i for i, c in enumerate(classes)}
    y = y_raw.map(mapping).to_numpy(dtype=np.int64)
    X_df = df.iloc[:, :-1]
    X_df = pd.get_dummies(X_df)
    X = X_df.to_numpy(dtype=np.float32)
    return X, y



def load_immunotherapy_dataset():
    data = np.loadtxt("data/immunotherapy.csv", delimiter=",", skiprows=1)
    X = data[:, :-1].astype(np.float32)
    y_raw = data[:, -1].astype(np.int64)
    classes = np.unique(y_raw)
    assert len(classes) == 2, "Immunotherapy dataset must be binary"
    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[c] for c in y_raw], dtype=np.int64)
    return X, y


def load_pima_diabetes_dataset():
    df = pd.read_csv("data/diabetes.csv")
    y = df["Outcome"].to_numpy(dtype=np.int64)
    classes = np.unique(y)
    assert set(classes) == {0, 1}, f"Pima must be binary, got {classes}"
    X = df.drop(columns=["Outcome"]).to_numpy(dtype=np.float32)
    return X, y



def load_adult_income_dataset():
    """
    Adult Census Income dataset (offline version).
    Binary classification: income <=50K (0) vs >50K (1)
    """
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Load local files
    train_data = pd.read_csv("data/adult.data", names=column_names, skipinitialspace=True)
    test_data = pd.read_csv("data/adult.test", names=column_names, skiprows=1, skipinitialspace=True)
    
    # Combine train and test
    df = pd.concat([train_data, test_data], ignore_index=True)
    
    # Remove rows with missing values (indicated by '?')
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    # Encode categorical variables
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                        'relationship', 'race', 'sex', 'native-country']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Encode target: income <=50K -> 0, >50K -> 1
    df['income'] = df['income'].astype(str).str.strip().str.replace('.', '', regex=False)
    y = df['income'].map({'<=50K': 0, '>50K': 1}).to_numpy(dtype=np.int64)
    
    # Drop target column
    X = df.drop(columns=['income']).to_numpy(dtype=np.float32)
    
    return X, y


def load_bank_marketing_dataset():
    """
    Bank Marketing dataset (offline version).
    Binary classification: no (0) vs yes (1) for term deposit subscription
    """
    df = pd.read_csv("data/bank-full.csv", sep=';')
    
    # Encode target: 'no' -> 0, 'yes' -> 1
    y = df['y'].map({'no': 0, 'yes': 1}).to_numpy(dtype=np.int64)
    
    # Drop target and duration (highly predictive for test data)
    X_df = df.drop(columns=['y', 'duration'] if 'duration' in df.columns else ['y'])
    
    # Encode categorical variables
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=True)
    
    X = X_df.to_numpy(dtype=np.float32)
    
    return X, y


def load_car_evaluation_dataset():
    """
    Car Evaluation dataset (offline version).
    Multi-class classification: acceptability levels
    Classes: unacc (0), acc (1), good (2), vgood (3)
    """
    column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    
    df = pd.read_csv("data/car.data", names=column_names)
    
    # Encode categorical features
    categorical_cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Encode target: class values
    class_mapping = {
        'unacc': 0,
        'acc': 1,
        'good': 2,
        'vgood': 3
    }
    y = df['class'].map(class_mapping).to_numpy(dtype=np.int64)
    
    X = df.drop(columns=['class']).to_numpy(dtype=np.float32)
    
    return X, y


def load_breast_cancer_wisconsin_dataset():
    """
    Breast Cancer Wisconsin (Diagnostic) dataset.
    Binary classification: Malignant (0) vs Benign (1)
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)  # 0=malignant, 1=benign
    return X, y

def load_credit_card_clients_dataset():
    """
    Default of Credit Card Clients dataset (offline version).
    Binary classification: default payment (0=no, 1=yes)
    """
    # Load Excel file
    df = pd.read_excel("data/default_of_credit_card_clients.xls", header=1)
    
    # First column is ID, drop it
    X = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    y = df.iloc[:, -1].to_numpy(dtype=np.int64)
    
    return X, y


# ---------------------------
# Multi-class classification datasets
# ---------------------------

def load_iris_dataset():
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    return X, y

def load_thyroid_dataset():
    data = np.loadtxt("data/thyroid.csv", delimiter=",", skiprows=1)

    X = data[:, :-1].astype(np.float32)

    y_raw = data[:, -1].astype(np.int64)

    classes = np.unique(y_raw)
    mapping = {c: i for i, c in enumerate(classes)}

    y = np.array([mapping[c] for c in y_raw], dtype=np.int64)

    return X, y


def load_wine_dataset():
    data = load_wine()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)
    return X, y


def encode_dna_sequence(seq):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    one_hot = np.zeros((len(seq), 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in mapping:
            one_hot[i, mapping[ch]] = 1.0
    return one_hot.flatten()



def load_digits_dataset(normalize=True):
    """
    Digits dataset (8x8 handwritten digits), multiclass classification.
    Returns flattened images of shape (N, 64).
    """
    data = load_digits()
    X = data.images.reshape(len(data.images), -1).astype(np.float32)
    y = data.target.astype(np.int64)

    if normalize:
        # Digits pixels are in range [0, 16]
        X /= 16.0

    return X, y


# ---------------------------
# Loader function
# ---------------------------

def load_dataset(name, test_size=0.2, random_state=42):

    # ---------------------------
    # Synthetic datasets
    # ---------------------------
    if name == "spiral":
        X, y = make_spirals()
        task_type = "classification"
        n_outputs = 1

    elif name == "circles":
        X, y = make_circles_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "moons":
        X, y = make_moons_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "swiss-roll":
        X, y = make_swiss_roll_dataset()
        task_type = "regression"
        n_outputs = 1

    elif name == "mackey-glass":
        X, y = generate_mackey_glass()
        task_type = "regression"
        n_outputs = 1

    # ---------------------------
    # Binary classification datasets
    # ---------------------------
    elif name == "haberman":
        X, y = load_haberman_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "cryotherapy":
        X, y = load_cryotherapy_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "heart":
        X, y = load_heart_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "autism":
        X, y = load_autism_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "immunotherapy":
        X, y = load_immunotherapy_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "pima":
        X, y = load_pima_diabetes_dataset()
        task_type = "classification"
        n_outputs = 1

    # ---------------------------
    # Multi-class classification datasets
    # ---------------------------
    elif name == "iris":
        X, y = load_iris_dataset()
        task_type = "classification"
        n_outputs = len(np.unique(y))  # 3

    elif name == "thyroid":
        X, y = load_thyroid_dataset()
        task_type = "classification"
        n_outputs = len(np.unique(y))

    elif name == "wine":
        X, y = load_wine_dataset()
        task_type = "classification"
        n_outputs = len(np.unique(y))

    elif name == "digits":
        X, y = load_digits_dataset()
        task_type = "classification"
        n_outputs = len(np.unique(y))  # 10

    elif name == "mnist":
        X_train, X_test, y_train, y_test = load_mnist_dataset()
        task_type = "classification"
        n_outputs = 10
        return X_train, X_test, y_train, y_test, task_type, n_outputs
    
    elif name == "breast-cancer-wisconsin":
        X, y = load_breast_cancer_wisconsin_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "adult-income":
        X, y = load_adult_income_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "bank-marketing":
        X, y = load_bank_marketing_dataset()
        task_type = "classification"
        n_outputs = 1

    elif name == "car-evaluation":
        X, y = load_car_evaluation_dataset()
        task_type = "classification"
        n_outputs = len(np.unique(y))  # 4 classes

    elif name == "credit-card":
        X, y = load_credit_card_clients_dataset()
        task_type = "classification"
        n_outputs = 1
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # ---------------------------
    # Train / test split
    # ---------------------------
    stratify = y if task_type in ["binary", "classification"] else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify
    )

    # Normalize feature values to prevent large raw logits and stabilize training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, task_type, n_outputs
