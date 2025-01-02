import json
from config import *
from torch_geometric.data import Data
    
def generateGrid(size=50, left_end=0.2, right_end=10):
    x = torch.linspace(left_end, right_end, size)
    y = torch.linspace(left_end, right_end, size)

    x, y = torch.meshgrid(x, y, indexing='ij')

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    grid = torch.cat((x, y), dim=1)
    
    return grid

edge_index_cache = []

def generateGridOne(size=50, left_end=0.2, right_end=10):
    x = torch.linspace(left_end, right_end, size)
    y = torch.linspace(left_end, right_end, size)

    x, y = torch.meshgrid(x, y, indexing='ij')

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    grid = torch.cat((x, y), dim=1)
    
    return grid

def generateEdgeIndex(grid, file_path=EDGE_INDEX_CACHE_NAME):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            edge_index_cache = json.load(file)
        return torch.tensor(edge_index_cache, dtype=torch.long).t().contiguous()
    
    edge_index_cache = []
    for i in range(len(grid)):
        for j in range(len(grid)):
            if torch.norm(grid[i] - grid[j]) < DIST_THRES:
                edge_index_cache.append([i, j])
    
    with open(file_path, "w") as file:
        json.dump(edge_index_cache, file)

    return torch.tensor(edge_index_cache, dtype=torch.long).t().contiguous()

class MyData:
    def __init__(self, feature, edge_index, label=None):
        self.feature = feature
        self.edge_index = edge_index
        self.label = label

def processFeature(grid, val, coordinate):
    features = []
    for i in range(coordinate.shape[0]):
        diff = grid - coordinate[i]
        norm = torch.norm(diff, dim=1)
        if RECIPROCAL:
            norm += 1e-9
            norm = 1 / norm

        #normalization
        min_val = torch.min(norm)
        max_val = torch.max(norm)
        
        normalized_dist = (norm - min_val) / (max_val - min_val)
        
        feature = torch.cat((normalized_dist.unsqueeze(1), val.unsqueeze(1)), dim=1)
        features.append(feature)

    return features

def generateTrainingData():
    grid1 = generateGrid(50, 0.2, 10)
    grid2 = generateGrid(10, 0.5, 9.5)
    
    if TWO_GRID:
        grid = torch.cat((grid1, grid2), dim=0)
    else:
        grid = generateGridOne()
    
    val1 = U(grid1[:, 0], grid1[:, 1])
    val2 = U(grid2[:, 0], grid2[:, 1])

    if USE_OBS:
        val1 = H1(U(grid1[:, 0], grid1[:, 1]))
        val2 = H2(U(grid2[:, 0], grid2[:, 1]))
        
    if TWO_GRID:
        val = torch.cat((val1, val2), dim=0)
    else:
        val = val1

    edge_index = generateEdgeIndex(grid)
    
    coordinate = grid.clone()
    feature = processFeature(grid, val, coordinate)

    padding1 = torch.zeros_like(val1)
    label1 = torch.stack((val1, padding1), dim=1)

    padding2 = torch.ones_like(val2)
    label2 = torch.stack((val2, padding2), dim=1)
    if TWO_GRID:
        label = torch.cat((label1, label2), dim=0)
    else:
        label = label1
        
    training_data = []
    
    for i in range(coordinate.shape[0]):
        training_data.append(MyData(feature[i], edge_index, label[i]))

    return training_data

def generateTestingData():
    grid1 = generateGrid(50, 0.2, 10)
    grid2 = generateGrid(10, 0.5, 9.5)
    
    if TWO_GRID:
        grid = torch.cat((grid1, grid2), dim=0)
    else:
        grid = generateGridOne()
    
    val1 = U(grid1[:, 0], grid1[:, 1])
    val2 = U(grid2[:, 0], grid2[:, 1])

    if USE_OBS:
        val1 = H1(U(grid1[:, 0], grid1[:, 1]))
        val2 = H2(U(grid2[:, 0], grid2[:, 1]))
        
    if TWO_GRID:
        val = torch.cat((val1, val2), dim=0)
    else:
        val = val1
        
    edge_index = generateEdgeIndex(grid)

    # x = torch.linspace(0, 10, 201)
    # y = torch.linspace(0, 10, 201)
    
    # x, y = torch.meshgrid(x, y, indexing='ij')

    # coordinate = torch.stack((x, y), dim=1)
    coordinate = generateGrid(60, 0, 10)

    feature = processFeature(grid, val, coordinate)

    testing_data = []

    for i in range(coordinate.shape[0]):
        testing_data.append(MyData(feature[i], edge_index))
        
    return testing_data, coordinate

def prepareForPlot(model, data, coordinate):
    real_val = U(coordinate[:, 0], coordinate[:, 1])
    predict_val = []
    for i in range(len(data)):
        predict_val.append(model(data[i]))
    predict_val = torch.stack(predict_val)
    coordinate_np = coordinate.detach().numpy()
    real_val_np = real_val.detach().numpy()
    predict_val_np = predict_val.detach().numpy()
    
    return coordinate_np, real_val_np, predict_val_np