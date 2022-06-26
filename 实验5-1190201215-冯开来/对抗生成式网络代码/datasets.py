import mat4py
import torch
from torch.utils.data import Dataset
import numpy as np


class Points(Dataset):
    def __init__(self):
        self.data = mat4py.loadmat("./dataset/points.mat")['xx']

    def __getitem__(self, idx):
        xy = torch.tensor(np.array(self.data[idx])).to(torch.float32)
        return xy

    def __len__(self):
        return len(self.data)
