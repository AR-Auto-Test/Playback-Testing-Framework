import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import pandas as pd
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, mat_file, metadata_dir):
        self.data = loadmat(mat_file)['frames']
        self.metadata_files = [os.path.join(metadata_dir, f) for f in os.listdir(metadata_dir) if f.endswith('.txt')]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        frames = self.data[idx]
        metadata = pd.read_csv(self.metadata_files[idx], header=None).values
        labels = metadata[:, -1]  # 假设标签在最后一列
        metadata = metadata[:, :-1]  # 其他列是metadata
        return torch.tensor(frames, dtype=torch.float), torch.tensor(metadata, dtype=torch.float), torch.tensor(labels, dtype=torch.long)