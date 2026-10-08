import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from config import *
from model import MLP

def plot_loss():
    loss_history = np.load(os.path.join(RESULTS_DIR, 'loss_history.npy'))
    train_loss = loss_history[0]
    val_loss = loss_history[1]
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'loss_vs_epoch.png'), dpi=300)
    plt.close()

def plot_test_samples():
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    Y_test = np.load(os.path.join(DATA_DIR, 'Y_test.npy'))
    wavelengths = np.load(os.path.join(DATA_DIR, 'wavelengths.npy'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'mlp_best.pth'), map_location=device))
    model.eval()
    
    indices = [0, 1, 2]
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        d = X_test[idx]
        true_spec = Y_test[idx]
        d_tensor = torch.tensor(d, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_spec = model(d_tensor).cpu().numpy().flatten()
        plt.subplot(1, 3, i + 1)
        plt.plot(wavelengths, true_spec, 'b-', label='TMM')
        plt.plot(wavelengths, pred_spec, 'r--', label='MLP')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Reflectance')
        plt.title(f'Test Sample {idx}\nd={d.round(1)}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'test_samples.png'), dpi=300)
    plt.close()

def plot_train_size():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'train_size_results.csv'))
    plt.figure(figsize=(8, 5))
    plt.plot(df['train_size'], df['test_mse'], 'o-')
    plt.xlabel('Training Set Size')
    plt.ylabel('Test MSE')
    plt.yscale('log')
    plt.title('Effect of Training Set Size on Test Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'train_size_effect.png'), dpi=300)
    plt.close()

def plot_design():
    top5_candidates = np.load(os.path.join(RESULTS_DIR, 'top5_candidates.npy'))
    top5_mlp_spectra = np.load(os.path.join(RESULTS_DIR, 'top5_mlp_spectra.npy'))
    top5_tmm_spectra = np.load(os.path.join(RESULTS_DIR, 'top5_tmm_spectra.npy'))
    wavelengths = np.load(os.path.join(DATA_DIR, 'wavelengths.npy'))
    
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, top5_mlp_spectra[0], 'r--', label='MLP Prediction')
    plt.plot(wavelengths, top5_tmm_spectra[0], 'b-', label='TMM Verification')
    plt.axvline(x=TARGET_WAVELENGTH, color='k', linestyle=':', label=f'Target {TARGET_WAVELENGTH} nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'Top1 Design Spectrum\nd={top5_candidates[0].round(1)}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'top1_design_spectrum.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 5))
    for i in range(len(top5_candidates)):
        plt.plot(wavelengths, top5_tmm_spectra[i], label=f'Top{i+1}')
    plt.axvline(x=TARGET_WAVELENGTH, color='k', linestyle=':', label=f'Target {TARGET_WAVELENGTH} nm')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title('Top5 Designs TMM Spectra')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'top5_designs_spectra.png'), dpi=300)
    plt.close()

def plot_failure_case():
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    Y_test = np.load(os.path.join(DATA_DIR, 'Y_test.npy'))
    wavelengths = np.load(os.path.join(DATA_DIR, 'wavelengths.npy'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'mlp_best.pth'), map_location=device))
    model.eval()
    
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_all = model(X_tensor).cpu().numpy()
    mse_all = np.mean((pred_all - Y_test) ** 2, axis=1)
    worst_idx = np.argmax(mse_all)
    
    d = X_test[worst_idx]
    true_spec = Y_test[worst_idx]
    pred_spec = pred_all[worst_idx]
    
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths, true_spec, 'b-', label='TMM')
    plt.plot(wavelengths, pred_spec, 'r--', label='MLP')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.title(f'Worst Test Sample (MSE={mse_all[worst_idx]:.2e})\nd={d.round(1)}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'failure_case.png'), dpi=300)
    plt.close()

def main():
    plot_loss()
    plot_test_samples()
    plot_train_size()
    plot_design()
    plot_failure_case()
    print("All figures generated.")

if __name__ == '__main__':
    main()