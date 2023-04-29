"""
Load custom dataset
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, df:pd.DataFrame, img_dir:str, transform=None, train=True):
        """
        Arguments:
             df: data frame contains image name and label.
             img_dir: path for images folder.
             transform: transform object contains all preprocessing for image.
             train: this is to distinguish between training data and inference
        Return:
            tuple contains image and label

        """
        self.img_labels = df
        self.img_dir = img_dir
        self.transform = transform
        self.train = train

    def __len__(self)->int:
        return len(self.img_labels)

    def __getitem__(self, idx:int)->tuple:
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        if self.train:
            label = self.img_labels.iloc[idx, 1]
            return image, label

        return image

