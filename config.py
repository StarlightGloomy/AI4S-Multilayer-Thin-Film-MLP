import os
import random
import numpy as np
import torch

# ==================== 个性化参数 ====================
# 学号后两位 N = 59, 59 mod 31 = 28
# lambda_target = 450 + 10 * (N mod 31) = 450 + 280 = 730 nm
TARGET_WAVELENGTH = 730.0  # nm
SEED = 270159
DESIGN_SEED = SEED + 1     # 270160

# ==================== 光学参数 ====================
N_H = 2.30
N_L = 1.45
N_GLASS = 1.52
N_AIR = 1.0

# 波长范围 400-800 nm，步长 10 nm，共 41 点
WAVELENGTH_START = 400
WAVELENGTH_END = 800
WAVELENGTH_STEP = 10
WAVELENGTHS = np.arange(WAVELENGTH_START, WAVELENGTH_END + 1, WAVELENGTH_STEP)

# 膜厚范围 40-180 nm
THICKNESS_MIN = 40.0
THICKNESS_MAX = 180.0

# ==================== 数据集参数 ====================
TOTAL_SIZE = 5000
TRAIN_SIZE = 4000
VAL_SIZE = 500
TEST_SIZE = 500
TRAIN_SIZES = [500, 1000, 2000, 4000]

# ==================== MLP 参数 ====================
INPUT_DIM = 4
OUTPUT_DIM = len(WAVELENGTHS)  # 41
MLP_HIDDEN = [128, 128, 64]
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 1e-3

# ==================== 设计筛选参数 ====================
NUM_CANDIDATES = 10000
TOP_K = 10
FINAL_TOP = 5

# ==================== 路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(RESULTS_DIR, 'models')
FIGURE_DIR = os.path.join(RESULTS_DIR, 'figures')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ==================== 随机种子 ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)