import pandas as pd
from sklearn import datasets
from ucimlrepo import fetch_ucirepo
from . import Test


class Digits_UCI(Test):
    def __init__(self, *args, **kwargs) -> None:
        data = fetch_ucirepo(id=81)

        self.df = data.data.features
        self.target = data.data.targets

        super().__init__(*args, **kwargs)


class Digits(Test):
    def __init__(self, *args, **kwargs) -> None:
        train_path = "./data/unfis_data/optdigits.tra"
        self.df = pd.read_csv(train_path, header=None)

        test_path = "./data/unfis_data/optdigits.tes"
        test_data = pd.read_csv(test_path, header=None)

        self.target = 64
        super().__init__(*args, **kwargs, test_data=test_data, index=False, train_size=3823)


class Segmentation(Test):
    def __init__(self, *args, **kwargs) -> None:
        df1 = "./data/unfis_data/segmentation.data"
        df2 = "./data/unfis_data/segmentation.test"
        
        df1 = pd.read_csv(df1, header=None)
        print(df1.shape)
        df2 = pd.read_csv(df2, header=None)
        print(df2.shape)

        self.df = pd.concat([df1, df2], axis=0)           
        print(self.df.shape)

        self.df = self.df.reset_index(drop=True)
        self.df = self.df.drop(1, axis=1)

        self.target = 0
        super().__init__(*args, **kwargs)


class Diabetes(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/diabetes/diabetes.csv"
        self.df = pd.read_csv(path)
        super().__init__(*args, **kwargs)


class BCW(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/unfis_data/breast-cancer-wisconsin.data"
        self.df = pd.read_csv(path, header=None)
        self.df.drop(0, axis=1, inplace=True)
        self.target=9
        print(self.df.shape)
        super().__init__(*args, **kwargs)

class DNA(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "./data/unfis_data/dna.arff"
        self.df = pd.read_csv(path, header=None)
        self.target=180
        print(self.df.shape)
        super().__init__(*args, **kwargs)

class Smoke(Test):
    def __init__(self, *args, **kwargs) -> None:
        path = "data/smoke/smoke_detection_iot.csv"
        self.df = pd.read_csv(path)
        self.df.drop(['index','UTC'], axis=1, inplace=True)
        self.target="Fire Alarm"
        print(self.df.shape)
        super().__init__(*args, **kwargs)

class MNIST(Test):
    def __init__(self, *args, **kwargs) -> None:
        train_path = "data/mnist/mnist_train.csv"
        self.df = pd.read_csv(train_path)

        test_path = "data/mnist/mnist_test.csv"
        test_data = pd.read_csv(test_path)

        self.target = 'label'
        
        super().__init__(*args, **kwargs, test_data=test_data, index=False)


class ORL(Test):
    def __init__(self, *args, **kwargs) -> None:
        import os

        def load_split(split_dir: str) -> pd.DataFrame:
            files = [f for f in os.listdir(split_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            rows, labels = [], []
            ref_size = None  # keep a consistent size across all images

            for fname in files:
                label = fname.rsplit("_", 1)[-1].split(".")[0]

                img = Image.open(os.path.join(split_dir, fname)).convert("L")
                if ref_size is None:
                    # use the first image as the reference size (H, W)
                    ref_size = img.size[::-1]  # (H, W)

                # ensure consistent size across the split
                img = img.resize(ref_size[::-1], Image.BILINEAR)

                arr = np.asarray(img, dtype=np.uint8).reshape(-1)  # flatten to 1D
                rows.append(arr)
                labels.append(label)

            df = pd.DataFrame(rows, columns=[f"p{i}" for i in range(len(rows[0]))])
            df["label"] = labels
            return df

        train_dir = "./data/orl/train"
        test_dir = "./data/orl/test"

        self.df = load_split(train_dir)
        test_data = load_split(test_dir)
        print(self.df)
        self.target = "label"
        super().__init__(task_type="classification", *args, **kwargs, test_data=test_data, index=False)


class FashionMNIST(Test):
    def __init__(self, *args, **kwargs) -> None:
        from tensorflow import keras
        import pandas as pd

        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

        # flatten 28x28 images to match your other tabular datasets
        train_df = pd.DataFrame(x_train.reshape(x_train.shape[0], -1))
        train_df["label"] = y_train

        test_df = pd.DataFrame(x_test.reshape(x_test.shape[0], -1))
        test_df["label"] = y_test

        self.df = train_df
        self.target = "label"

        super().__init__(task_type="classification", *args, **kwargs, test_data=test_df, index=False)