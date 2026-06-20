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

