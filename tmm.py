import numpy as np
from config import N_H, N_L, N_GLASS, N_AIR

def tmm_reflectance(thicknesses, wavelengths):
    """
    计算多层膜反射率。
    thicknesses: [d1, d2, d3, d4] 对应 H, L, H, L，单位 nm
    wavelengths: 波长数组，单位 nm
    返回: 反射率数组，与 wavelengths 同形状
    """
    n_layers = [N_H, N_L, N_H, N_L]
    num_layers = len(n_layers)
    R = np.zeros_like(wavelengths, dtype=float)
    
    for idx, lam in enumerate(wavelengths):
        M = np.eye(2, dtype=complex)
        for i in range(num_layers):
            n = n_layers[i]
            d = thicknesses[i]
            delta = 2 * np.pi * n * d / lam
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            eta = n  # 正入射
            m_i = np.array([[cos_d, 1j * sin_d / eta],
                            [1j * eta * sin_d, cos_d]], dtype=complex)
            M = M @ m_i
        
        ns = N_GLASS
        n0 = N_AIR
        M11, M12 = M[0, 0], M[0, 1]
        M21, M22 = M[1, 0], M[1, 1]
        numerator = n0 * M11 + n0 * ns * M12 - M21 - ns * M22
        denominator = n0 * M11 + n0 * ns * M12 + M21 + ns * M22
        r = numerator / denominator
        R[idx] = np.abs(r) ** 2
    
    return R