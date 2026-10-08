import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from config import *
from model import MLP
from dataset import get_dataloaders
from utils import set_seed

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            total_loss += loss.item()
            total_samples += X_batch.size(0)
    return total_loss / total_samples

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    for train_size in TRAIN_SIZES:
        print(f"\nTraining with {train_size} samples...")
        set_seed(SEED)
        train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE, train_size)
        model = MLP().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        best_val_loss = float('inf')
        for epoch in range(EPOCHS):
            model.train()
            for X_batch, Y_batch in train_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, Y_batch in val_loader:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, Y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        test_mse = evaluate_model(model, test_loader, device)
        print(f"Train size: {train_size}, Test MSE: {test_mse:.6e}")
        results.append({'train_size': train_size, 'test_mse': test_mse})
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'train_size_results.csv'), index=False)
    print("Training size experiment completed.")

if __name__ == '__main__':
    run_experiment()