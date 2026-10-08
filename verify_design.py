import numpy as np
import pandas as pd
import os
from config import *
from tmm import tmm_reflectance

def verify_candidates():
    top10_candidates = np.load(os.path.join(RESULTS_DIR, 'top10_candidates.npy'))
    top10_pred_spectra = np.load(os.path.join(RESULTS_DIR, 'top10_pred_spectra.npy'))
    top10_mlp_target = np.load(os.path.join(RESULTS_DIR, 'top10_mlp_target.npy'))
    
    target_idx = np.argmin(np.abs(WAVELENGTHS - TARGET_WAVELENGTH))
    
    top10_tmm_spectra = np.zeros((len(top10_candidates), len(WAVELENGTHS)))
    top10_tmm_target = np.zeros(len(top10_candidates))
    for i, d in enumerate(top10_candidates):
        spec = tmm_reflectance(d, WAVELENGTHS)
        top10_tmm_spectra[i] = spec
        top10_tmm_target[i] = spec[target_idx]
    
    sorted_indices = np.argsort(top10_tmm_target)[::-1]
    top5_indices = sorted_indices[:FINAL_TOP]
    
    top5_candidates = top10_candidates[top5_indices]
    top5_mlp_target = top10_mlp_target[top5_indices]
    top5_tmm_target = top10_tmm_target[top5_indices]
    top5_pred_spectra = top10_pred_spectra[top5_indices]
    top5_tmm_spectra = top10_tmm_spectra[top5_indices]
    
    df_top5 = pd.DataFrame(top5_candidates, columns=['d1', 'd2', 'd3', 'd4'])
    df_top5['MLP_Target'] = top5_mlp_target
    df_top5['TMM_Target'] = top5_tmm_target
    df_top5.to_csv(os.path.join(RESULTS_DIR, 'top5_designs.csv'), index=False)
    
    np.save(os.path.join(RESULTS_DIR, 'top5_candidates.npy'), top5_candidates)
    np.save(os.path.join(RESULTS_DIR, 'top5_mlp_spectra.npy'), top5_pred_spectra)
    np.save(os.path.join(RESULTS_DIR, 'top5_tmm_spectra.npy'), top5_tmm_spectra)
    np.save(os.path.join(RESULTS_DIR, 'top5_mlp_target.npy'), top5_mlp_target)
    np.save(os.path.join(RESULTS_DIR, 'top5_tmm_target.npy'), top5_tmm_target)
    
    print("Verification completed. Top5 designs:")
    print(df_top5)

if __name__ == '__main__':
    verify_candidates()