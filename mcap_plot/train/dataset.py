import pandas as pd
import numpy
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, txt):
        df = pd.read_csv(txt, sep=" ", header=None, names=["distance", "intensity", "cosine_a", "cosine_b"])
        self.data = torch.tensor(df.iloc[:, [0, 2, 3]].values, dtype=torch.float32)
        self.intensity = torch.tensor(df.iloc[:, [1]].values, dtype=torch.float32)
    
    def __getitem__(self, index):
        return self.data[index], self.intensity[index]

    def __len__(self):
        return len(self.data)