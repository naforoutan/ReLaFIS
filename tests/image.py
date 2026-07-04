"""
Image and vision datasets.

All datasets here produce tabular DataFrames of pixel values; the Test base
class sees them exactly like any other tabular dataset.  The heavy
framework-specific imports (tensorflow, PIL) are deferred into __init__ so
they don't break the module import when those libraries are absent.

Pre-defined train/test files are merged into one pool, then split 70/30
via tests.base.Test.
"""

import pandas as pd
from .base import Test


class MNIST(Test):
    """MNIST handwritten digits loaded from local CSV files."""

    def __init__(self, *args, **kwargs) -> None:
        train_df = pd.read_csv("data/mnist/mnist_train.csv")
        test_df = pd.read_csv("data/mnist/mnist_test.csv")
        self.df = pd.concat([train_df, test_df], ignore_index=True)
        self.target = "label"
        super().__init__(*args, **kwargs, index=False)


class FashionMNIST(Test):
    """Fashion-MNIST loaded via tf.keras (requires TensorFlow)."""

    def __init__(self, *args, **kwargs) -> None:
        from tensorflow import keras

        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        train_df = pd.DataFrame(x_train.reshape(x_train.shape[0], -1))
        train_df["label"] = y_train

        test_df = pd.DataFrame(x_test.reshape(x_test.shape[0], -1))
        test_df["label"] = y_test

        self.df = pd.concat([train_df, test_df], ignore_index=True)
        self.target = "label"
        super().__init__(task_type="classification", *args, **kwargs, index=False)
