import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from config import DATA_DIR, BATCH_SIZE

class ThinFilmDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataloaders(batch_size=BATCH_SIZE, train_size=None):
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    Y_train = np.load(os.path.join(DATA_DIR, 'Y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    Y_val = np.load(os.path.join(DATA_DIR, 'Y_val.npy'))
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    Y_test = np.load(os.path.join(DATA_DIR, 'Y_test.npy'))
    
    if train_size is not None:
        X_train = X_train[:train_size]
        Y_train = Y_train[:train_size]
    
    train_dataset = ThinFilmDataset(X_train, Y_train)
    val_dataset = ThinFilmDataset(X_val, Y_val)
    test_dataset = ThinFilmDataset(X_test, Y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader