import os
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import convert_image_dtype
from torch.utils.data import Dataset

class RuHandwrittenDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = convert_image_dtype(image, dtype=torch.float32)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
