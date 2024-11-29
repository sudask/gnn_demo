from config import *
from torch_geometric.data import Data
    
def generateFeature(size, left_end=0, right_end=10, drop_first=True):
    x = torch.linspace(left_end, right_end, size+1)
    y = torch.linspace(left_end, right_end, size+1)

    if drop_first:
        x = x[1:]
        y = y[1:]

    x, y = torch.meshgrid(x, y, indexing='ij')

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    grid = torch.cat((x, y), dim=1)
    val = U(grid[:, 0], grid[:, 1])

    return torch.cat((grid, val.unsqueeze(1)), dim=1)

def generateEdgeIndex(size):
    edge_index = []
    for i in range(size * size):
        if i % size < size - 1:
            edge_index.append([i, i + 1])
        if i < size * (size - 1):
            edge_index.append([i, i + size])

    return edge_index

def packData(feature, edge_index, coordinate):
    padding = torch.zeros((coordinate.shape[0], 1))
    padded_coordinate = torch.cat((coordinate, padding), dim=1)

    function_val = U(coordinate[:, 0], coordinate[:, 1])

    size = feature.shape[0]
    for i in range(size):
        edge_index.append([i, size])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    feature_plus_coordinate = []
    for i in range(padded_coordinate.shape[0]):
        feature_plus_coordinate.append(torch.cat((feature, padded_coordinate[i].unsqueeze(0)), dim=0))

    data = []
    for i in range(len(feature_plus_coordinate)):
        data.append(Data(x=feature_plus_coordinate[i], edge_index=edge_index, y=function_val[i]))

    return data
    
def generateTrainingData(size):
    feature = generateFeature(size)
    edge_index = generateEdgeIndex(size)
    
    x = torch.linspace(1, 10, size)
    y = torch.linspace(1, 10, size)

    x, y = torch.meshgrid(x, y, indexing='ij')

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    coordinate = torch.cat((x, y), dim=1)

    return packData(feature, edge_index, coordinate)

if __name__ == "__main__":
    print(generateTrainingData(10))