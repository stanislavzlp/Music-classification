"""
Определим генераторы данных для обучения и валидации модели
"""

from torch.utils.data import Dataset, DataLoader
from glob import glob
import pandas as pd
import numpy as np
import torch


class SpectralDataset(Dataset):
    def __init__(self, root_dir: str, genres: str):
        self.root_dir = root_dir
        genres_df = pd.read_csv(genres)
        genres_df = genres_df.drop(columns=['song'])
        self.genres = np.array(genres_df)
        self.image_paths = glob(self.root_dir + '/*.npy')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = np.load(self.image_paths[item])
        image = torch.from_numpy(image[np.newaxis, ...]).type(torch.float32)

        id = int(self.image_paths[item].split('/')[-1].split('_')[0])  # id трека берём из названия файла
        label = self.genres[id]
        label = torch.from_numpy(label).type(torch.float32)
        return image, label


train_dataset = SpectralDataset(root_dir='Data/Patches/Train', genres='Data/genres.csv')
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

val_dataset = SpectralDataset(root_dir='Data/Patches/Validation', genres='Data/genres.csv')
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
