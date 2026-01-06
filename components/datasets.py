import numpy as np
from sklearn.datasets import make_moons, make_circles, make_swiss_roll, load_iris
from sklearn.model_selection import train_test_split

# ---------------------------
# Synthetic datasets
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

def make_swiss_roll_dataset(n_samples=1000, noise=0.1):
    """3D swiss roll dataset, regression on unrolled coordinate"""
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise)
    # Optional: normalize t to [0,1] as regression target
    t = (t - t.min()) / (t.max() - t.min())
    return X.astype(np.float32), t.astype(np.float32).reshape(-1,1)

# ---------------------------
# Real datasets
# ---------------------------

def load_iris_dataset():
    """Iris dataset, multiclass classification"""
    data = load_iris()
    X = data.data.astype(np.float32)
    y = data.target
    return X, y

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
# Loader function
# ---------------------------

def load_dataset(name, test_size=0.2, random_state=42):
    """
    Load a dataset by name and return train/test split.
    
    Returns:
        X_train, X_test, y_train, y_test, task_type, n_outputs
    """
    if name == "spiral":
        X, y = make_spirals()
        task_type = "binary"
        n_outputs = 1
    elif name == "circles":
        X, y = make_circles_dataset()
        task_type = "binary"
        n_outputs = 1
    elif name == "moons":
        X, y = make_moons_dataset()
        task_type = "binary"
        n_outputs = 1
    elif name == "swiss-roll":
        X, y = make_swiss_roll_dataset()
        task_type = "regression"
        n_outputs = 1
    elif name == "iris":
        X, y = load_iris_dataset()
        task_type = "multiclass"
        n_outputs = len(np.unique(y))
    elif name == "mackey-glass":
        X, y = generate_mackey_glass()
        task_type = "regression"
        n_outputs = 1
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    return X_train, X_test, y_train, y_test, task_type, n_outputs
