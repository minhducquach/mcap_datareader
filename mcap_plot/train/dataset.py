import pandas as pd
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, txt):
        df = pd.read_csv(txt, sep=" ", header=None, names=["distance", "intensity", "cosine_a", "cosine_b"])
        
        # Min-Max Normalization for the features (distance, cosine_a, cosine_b)
        features = df.iloc[:, [0, 2, 3]].values
        print(features[:, 2])
        features[:, 2] = abs(features[:, 2])
        self.intensity = torch.tensor(df.iloc[:, [1]].values, dtype=torch.float32)
        
        # Compute min and max values for each feature column
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        
        # Apply Min-Max normalization
        normalized_features = (features - min_vals) / (max_vals - min_vals)
        
        # Convert normalized features to a tensor
        self.data = torch.tensor(normalized_features, dtype=torch.float32)
    
    def __getitem__(self, index):
        return self.data[index], self.intensity[index]

    def __len__(self):
        return len(self.data)
