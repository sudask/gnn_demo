from config import *
from torch_geometric.data import Data
    
def generateGrid(size, left_end=0, right_end=10, drop_first=True):
    x = torch.linspace(left_end, right_end, size+1)
    y = torch.linspace(left_end, right_end, size+1)

    if drop_first:
        x = x[1:]
        y = y[1:]

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

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def packData(grid, val, edge_index, coordinate):
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
    
def generateTrainingData(size):
    grid = generateGrid(size)
    val = H1(U(grid[:, 0], grid[:, 1])).unsqueeze(1)
    edge_index = generateEdgeIndex(size)
    
    x = torch.linspace(1, 10, size)
    y = torch.linspace(1, 10, size)

    x, y = torch.meshgrid(x, y, indexing='ij')

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    coordinate = torch.cat((x, y), dim=1)

    return packData(grid, val, edge_index, coordinate)

if __name__ == "__main__":
    print(generateTrainingData(10))