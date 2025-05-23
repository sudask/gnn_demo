import xarray as xr  
import numpy as np

def generateEdgeIndex(obs_station):
    edge_index = []
    for i in range(len(obs_station)):
        for j in range(len(obs_station)):
            if np.linalg.norm(obs_station[i] - obs_station[j]) < 1.0:
                edge_index.append([i, j])
    
    return np.array(edge_index).T

def generateEdgeIndexAndWeight(obs_station):
    edge_index = []
    edge_weight = []
    for i in range(len(obs_station)):
        for j in range(len(obs_station)):
            distance = np.linalg.norm(obs_station[i] - obs_station[j])
            if distance < 1.5:
                edge_index.append([i, j])
                edge_weight.append(1.0 / (distance + 1e-5))  # 避免除以0
                
    edge_index = np.array(edge_index).T
    edge_weight = np.array(edge_weight)
    return edge_index, edge_weight