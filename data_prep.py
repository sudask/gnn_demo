from config import *
from torch_geometric.data import Data
    
def generateGrid(size, left_end=1, right_end=10):
    x = torch.linspace(left_end, right_end, size)
    y = torch.linspace(left_end, right_end, size)

    x, y = torch.meshgrid(x, y, indexing='ij')

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    grid = torch.cat((x, y), dim=1)
    
    return grid

def generateEdgeIndex(size):
    edge_index = []
    for i in range(size * size):
        if i % size < size - 1:
            edge_index.append([i, i + 1])
        if i < size * (size - 1):
            edge_index.append([i, i + size])
        if i % size > 0:
            edge_index.append([i, i - 1])
        if i >= size:
            edge_index.append([i, i - size])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def packDataSimp(grid, val, edge_index, coordinate):
    data = []
    U_val = U(coordinate[:, 0], coordinate[:, 1])
    for i in range(coordinate.shape[0]):
        diff = grid - coordinate[i]
        norm = torch.norm(diff, dim=1) + 1e-9
        norm = 1 / norm
        # normalization
        min_val = torch.min(norm)
        max_val = torch.max(norm)
        
        normalized_dist = (norm - min_val) / (max_val - min_val)
        
        feature = torch.cat((normalized_dist.unsqueeze(1), val), dim=1)
        data.append(Data(x=feature, edge_index=edge_index, y=U_val[i]))
    return data
    
def generateTrainingDataSimp(size):
    grid = generateGrid(size)
    val = H1(U(grid[:, 0], grid[:, 1])).unsqueeze(1)
    edge_index = generateEdgeIndex(size)
    
    x = torch.linspace(1, 10, size)
    y = torch.linspace(1, 10, size)

    x, y = torch.meshgrid(x, y, indexing='ij')

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    coordinate = torch.cat((x, y), dim=1)

    return packDataSimp(grid, val, edge_index, coordinate)

class MyData:
    def __init__(self, feature1, feature2, edge_index1, edge_index2, label=None):
        self.feature1 = feature1
        self.feature2 = feature2
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.label = label

def processFeature(grid, val, coordinate):
    features = []
    for i in range(coordinate.shape[0]):
        diff = grid - coordinate[i]
        norm = torch.norm(diff, dim=1) + 1e-9
        norm = 1 / norm
        #normalization
        min_val = torch.min(norm)
        max_val = torch.max(norm)
        
        normalized_dist = (norm - min_val) / (max_val - min_val)
        
        feature = torch.cat((normalized_dist.unsqueeze(1), val.unsqueeze(1)), dim=1)
        features.append(feature)

    return features

def generateTrainingData():
    grid1 = generateGrid(46)
    grid2 = generateGrid(10)

    # val1 = H1(U(grid1[:, 0], grid1[:, 1]))
    # val2 = H2(U(grid2[:, 0], grid2[:, 1]))
    
    val1 = U(grid1[:, 0], grid1[:, 1])
    val2 = U(grid2[:, 0], grid2[:, 1])

    edge_index1 = generateEdgeIndex(46)
    edge_index2 = generateEdgeIndex(10)
    
    coordinate1 = grid1.clone()
    coordinate2 = grid2.clone()

    feature11 = processFeature(grid1, val1, coordinate1)
    feature12 = processFeature(grid2, val2, coordinate1)

    padding1 = torch.zeros_like(val1)
    label1 = torch.stack((val1, padding1), dim=1)

    feature21 = processFeature(grid1, val1, coordinate2)
    feature22 = processFeature(grid2, val2, coordinate2)

    padding2 = torch.zeros_like(val2)
    label2 = torch.stack((val2, padding2), dim=1)

    training_data = []

    for i in range(coordinate1.shape[0]):
        training_data.append(MyData(feature11[i], feature12[i], edge_index1, edge_index2, label1[i]))

    for i in range(coordinate2.shape[0]):
        training_data.append(MyData(feature21[i], feature22[i], edge_index1, edge_index2, label2[i]))

    return training_data

def generateTestingData():
    grid1 = generateGrid(46)
    grid2 = generateGrid(10)

    # val1 = H1(U(grid1[:, 0], grid1[:, 1]))
    # val2 = H2(U(grid2[:, 0], grid2[:, 1]))
    
    val1 = U(grid1[:, 0], grid1[:, 1])
    val2 = U(grid2[:, 0], grid2[:, 1])

    edge_index1 = generateEdgeIndex(46)
    edge_index2 = generateEdgeIndex(10)

    x = torch.rand(300) * 10
    y = torch.rand(300) * 10

    coordinate = torch.stack((x, y), dim=1)

    feature1 = processFeature(grid1, val1, coordinate)
    feature2 = processFeature(grid2, val2, coordinate)

    testing_data = []

    for i in range(coordinate.shape[0]):
        testing_data.append(MyData(feature1[i], feature2[i], edge_index1, edge_index2))

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