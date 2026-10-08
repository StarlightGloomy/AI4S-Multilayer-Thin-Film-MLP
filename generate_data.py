import numpy as np
import os
from config import *
from tmm import tmm_reflectance

def generate_dataset():
    set_seed(SEED)
    thicknesses = np.random.uniform(THICKNESS_MIN, THICKNESS_MAX, size=(TOTAL_SIZE, 4))
    spectra = np.zeros((TOTAL_SIZE, len(WAVELENGTHS)))
    
    for i in range(TOTAL_SIZE):
        spectra[i] = tmm_reflectance(thicknesses[i], WAVELENGTHS)
        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1}/{TOTAL_SIZE}")
    
    X_train = thicknesses[:TRAIN_SIZE]
    Y_train = spectra[:TRAIN_SIZE]
    X_val = thicknesses[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    Y_val = spectra[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    X_test = thicknesses[TRAIN_SIZE + VAL_SIZE:]
    Y_test = spectra[TRAIN_SIZE + VAL_SIZE:]
    
    np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(DATA_DIR, 'Y_train.npy'), Y_train)
    np.save(os.path.join(DATA_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(DATA_DIR, 'Y_val.npy'), Y_val)
    np.save(os.path.join(DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(DATA_DIR, 'Y_test.npy'), Y_test)
    np.save(os.path.join(DATA_DIR, 'wavelengths.npy'), WAVELENGTHS)
    print("Data generation completed.")

if __name__ == '__main__':
    generate_dataset()