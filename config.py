import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random
import os

# Parameters
EPOCH = 100
TEST_NUM = 1000

# GRID_SIZE = 50
# LR = 0.002

# GRID_SIZE=20
# LR = 0.001

GRID_SIZE = 10
LR = 0.005

SAVE_DIR = "checkpoints"

def U(x, y):
    term1 = torch.sin(x + np.cos(x + y))
    term2 = 0.5 * torch.cos(x / 2 - y / 3) * torch.sin(y / 2 - x / 2)
    term3 = 0.1 * torch.sin(2 * x - 3 * y) * torch.cos(4 * y - 3 * x)
    return term1 + term2 - term3

def H1(u):
    return u ** 2 - u

def H2(u):
    return torch.cos(u - torch.sin(u))

def observeOperator(u):
    res = torch.zeros(u.shape[0])
    for i in range(u.shape[0]):
        if u[i, 1] == 0:
            res[i] = H1(u[i, 0])
        else:
            res[i] = H2(u[i, 0])