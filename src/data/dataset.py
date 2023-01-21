import gc

import pandas as pd
import torch
import torchvision as tvis
from loguru import logger
from PIL import Image
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Dataset, Subset

preprocessor = tvis.transforms.Compose(
    [
        tvis.transforms.ToTensor(),
        tvis.transforms.Resize((110, 110)),
        tvis.transforms.Normalize((0.5), (0.2)),
    ]
)


BAD_IMAGES = [
    "Abdullah_al-Attiyah_0001.jpg",
    "Dereck_Whittenburg_0001.jpg",
    "Lawrence_Foley_0001.jpg",
]


def train_test_split(dataset: Dataset, test_size: float):
    split_generator = ShuffleSplit(
        n_splits=1, test_size=test_size, random_state=0
    ).split(dataset)
    train_idx, test_idx = list(split_generator)[0]

    train_dataset = Subset(dataset=dataset, indices=train_idx)
    test_dataset = Subset(dataset=dataset, indices=test_idx)

    return train_dataset, test_dataset


class LFWDataset(Dataset):
    def __init__(self, path: str, preprocessed_prefix: str):
        self.data = []
        self.data_tuples = None
        self.path = path
        self.preprocessed_prefix = preprocessed_prefix
        self.on_cuda = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # returns x1, x2, y
        return self.data[idx]

    def filter_bad_images(self, df):
        return df[~df.img1_full_id.isin(BAD_IMAGES) & ~df.img2_full_id.isin(BAD_IMAGES)]

    def load_data(self):
        self.on_cuda = False
        self.load_df()
        self.load_images()

    def load_df(self):
        logger.info("Loading pairs dataframe")
        df = pd.read_parquet(self.path).pipe(self.filter_bad_images)

        df["prepro_img1"] = df.img1_full_id.apply(
            lambda s: f"{self.preprocessed_prefix}/{s}"
        )
        df["prepro_img2"] = df.img2_full_id.apply(
            lambda s: f"{self.preprocessed_prefix}/{s}"
        )

        self.data_tuples = df[["prepro_img1", "prepro_img2", "match"]].values.copy()
        del df

    def load_images(self):
        logger.info("Loading images")

        for im1_path, im2_path, match in self.data_tuples:
            im1 = preprocessor(Image.open(im1_path))
            im2 = preprocessor(Image.open(im2_path))
            self.data.append((im1, im2, torch.as_tensor(match).float()))

    def to_cuda(self):
        if self.on_cuda:
            return

        if not torch.cuda.is_available():
            raise Exception("CUDA isn't available.")

        logger.info("Loading dataset to CUDA")

        new_data = []

        for im1, im2, target in self.data:
            im1 = im1.to("cuda")
            im2 = im2.to("cuda")
            target = target.to("cuda")

            new_data.append((im1, im2, target))

        gc.collect()
        self.data = new_data
        self.on_cuda = True

    def free(self):
        for im1, im2, target in self.data:
            self.free_elements(im1, im2, target)
        self.data = []
        torch.cuda.empty_cache()
        gc.collect()

    def free_elements(self, im1, im2, target):
        im1.to("cpu")
        im2.to("cpu")
        target.to("cpu")

        del im1
        del im2
        del target


LFW_DATASET = LFWDataset(
    path="data/interim/pairs.parquet",
    preprocessed_prefix="data/preprocessed/images",
)
