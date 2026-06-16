"""
Image and vision datasets.

All datasets here produce tabular DataFrames of pixel values; the Test base
class sees them exactly like any other tabular dataset.  The heavy
framework-specific imports (tensorflow, PIL) are deferred into __init__ so
they don't break the module import when those libraries are absent.

To add a new image dataset:
  - Flatten images to 1-D feature vectors.
  - Store labels as a string or integer Series in self.target.
  - Pass test_data and index=False to super().__init__() for pre-split sets.
"""

import pandas as pd
from .base import Test

class MNIST(Test):
    """MNIST handwritten digits loaded from local CSV files."""

    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("data/mnist/mnist_train.csv")
        test_data = pd.read_csv("data/mnist/mnist_test.csv")
        self.target = "label"
        super().__init__(*args, **kwargs, test_data=test_data, index=False)


class FashionMNIST(Test):
    """Fashion-MNIST loaded via tf.keras (requires TensorFlow)."""

    def __init__(self, *args, **kwargs) -> None:
        from tensorflow import keras

        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        train_df = pd.DataFrame(x_train.reshape(x_train.shape[0], -1))
        train_df["label"] = y_train

        test_df = pd.DataFrame(x_test.reshape(x_test.shape[0], -1))
        test_df["label"] = y_test

        self.df = train_df
        self.target = "label"
        super().__init__(task_type="classification", *args, **kwargs, test_data=test_df, index=False)


class ORL(Test):
    """ORL (AT&T) face database loaded from train/test image directories.

    Expects images in ./data/orl/train/ and ./data/orl/test/, named
    <anything>_<label>.<ext>.  All images are converted to greyscale and
    flattened to a 1-D pixel vector.
    """

    def __init__(self, *args, **kwargs) -> None:
        import os
        import numpy as np
        from PIL import Image

        def load_split(split_dir: str) -> pd.DataFrame:
            files = [
                f for f in os.listdir(split_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            rows, labels = [], []
            ref_size = None

            for fname in files:
                label = fname.rsplit("_", 1)[-1].split(".")[0]
                img = Image.open(os.path.join(split_dir, fname)).convert("L")
                if ref_size is None:
                    ref_size = img.size[::-1]  # (H, W)
                img = img.resize(ref_size[::-1], Image.BILINEAR)
                rows.append(np.asarray(img, dtype=np.uint8).reshape(-1))
                labels.append(label)

            df = pd.DataFrame(rows, columns=[f"p{i}" for i in range(len(rows[0]))])
            df["label"] = labels
            return df

        self.df = load_split("./data/orl/train")
        test_data = load_split("./data/orl/test")
        self.target = "label"
        super().__init__(task_type="classification", *args, **kwargs, test_data=test_data, index=False)
