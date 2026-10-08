import numpy as np
import torch
import os
import pandas as pd
from config import *
from model import MLP
from utils import set_seed

def screen_candidates():
    set_seed(DESIGN_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MLP().to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'mlp_best.pth'), map_location=device))
    model.eval()
    
    candidates = np.random.uniform(THICKNESS_MIN, THICKNESS_MAX, size=(NUM_CANDIDATES, 4))
    candidates_tensor = torch.tensor(candidates, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_spectra = model(candidates_tensor).cpu().numpy()
    
    target_idx = np.argmin(np.abs(WAVELENGTHS - TARGET_WAVELENGTH))
    print(f"Target wavelength: {TARGET_WAVELENGTH} nm, index: {target_idx}, actual: {WAVELENGTHS[target_idx]} nm")
    
    target_reflectance = pred_spectra[:, target_idx]
    sorted_indices = np.argsort(target_reflectance)[::-1]  # 降序，寻找高反射
    top10_indices = sorted_indices[:TOP_K]
    top10_candidates = candidates[top10_indices]
    top10_pred_spectra = pred_spectra[top10_indices]
    top10_pred_target = target_reflectance[top10_indices]
    
    df_top10 = pd.DataFrame(top10_candidates, columns=['d1', 'd2', 'd3', 'd4'])
    df_top10['MLP_Target'] = top10_pred_target
    df_top10.to_csv(os.path.join(RESULTS_DIR, 'candidates_top10.csv'), index=False)
    
    np.save(os.path.join(RESULTS_DIR, 'top10_pred_spectra.npy'), top10_pred_spectra)
    np.save(os.path.join(RESULTS_DIR, 'top10_candidates.npy'), top10_candidates)
    np.save(os.path.join(RESULTS_DIR, 'top10_mlp_target.npy'), top10_pred_target)
    
    print("Top10 candidates saved.")
    return top10_candidates, top10_pred_spectra

if __name__ == '__main__':
    screen_candidates()