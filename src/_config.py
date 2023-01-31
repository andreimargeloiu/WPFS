BASE_DIR = '.' # path to the project directory

DATA_DIR = f'{BASE_DIR}/data'
LOGS_DIR = f'{BASE_DIR}/logs'
RESULTS_DIR = f"{BASE_DIR}/results"

SEED_VALUE = 42



import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed
