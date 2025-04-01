import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import os
from matplotlib import pyplot as plt

# Parameters
EPOCH = 100
# for reciprocal
# LEARING_RATE = 0.0005

# for one grid
# LEARING_RATE = 0.002

# for two grid
# LEARING_RATE = 0.002

# for full question
LEARING_RATE = 0.001

DIST_THRES = 0.3
SAVE_DIR = "checkpoints"

TWO_GRID = True
USE_OBS = True
RECIPROCAL = False
PTH_FILE_NAME = ("two_grid" if TWO_GRID else "one_grid") + ".pth"

TEST_NUM = 2000
EDGE_INDEX_CACHE_NAME = "edge_index/" + ("two_grid.json" if TWO_GRID else "one_grid.json")

def U(x, y):
    term1 = torch.sin(x + np.cos(x + y))
    term2 = 0.5 * torch.cos(x / 2 - y / 3) * torch.sin(y / 2 - x / 2)
    term3 = 0.1 * torch.sin(2 * x - 3 * y) * torch.cos(4 * y - 3 * x)
    return term1 + term2 - term3

# def H1(u):
#     return (u ** 2 - u) / 3

def H1(u):
    return u**3 / 4

# def H2(u):
#     return torch.cos(u - torch.sin(u))

def H2(u):
    return torch.cos(u / 2 + 4 - torch.sin(u / 2 + 4))

def observeOperator(u, label):
    if label == 0:
        return H1(u)
    
    if label == 1:
        return H2(u)