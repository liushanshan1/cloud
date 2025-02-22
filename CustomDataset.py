# File: ulti/CustomDataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class VibrationDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.data_dir = os.path.join(root_dir, mode)
        self.file_list = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npy')])
        self.transform = transform
        
        # 自动获取类别数
        self.classes = list({f.split('_')[0] for f in self.file_list})
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(self.classes))}
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        data = np.load(os.path.join(self.data_dir, file_name))
        
        # 标准化处理
        data = (data - data.mean()) / (data.std() + 1e-8)
        signal = torch.FloatTensor(data).unsqueeze(0)  # (1, seq_len)
        
        # 标签处理
        class_name = file_name.split('_')[0]
        label = self.class_to_idx[class_name]
        return signal, label