from config import *
from torch_geometric.data import Data

class GraphDataSet(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def generate_features():
    x = torch.linspace(1, GRID_SIZE, GRID_SIZE) / (GRID_SIZE / 10)
    y = torch.linspace(1, GRID_SIZE, GRID_SIZE) / (GRID_SIZE / 10)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    val = U(X, Y)
    feature = torch.stack((X.flatten(), Y.flatten(), val.flatten()), dim=1)
    edge = []
    n = GRID_SIZE
    for i in range(n * n):
        if i % n < n - 1:
            edge.append([i, i + 1])
        if i < n * (n - 1):
            edge.append([i, i + n])
    # edge = torch.tensor(edge, dtype=torch.long).t().contiguous()
    return feature.float(), edge

def prepare_training_data():
    feature, edge = generate_features()
    for i in range(GRID_SIZE * GRID_SIZE):
        edge.append([GRID_SIZE * GRID_SIZE, i])
    edge = torch.tensor(edge, dtype=torch.long).t().contiguous()
    
    x = torch.linspace(1, GRID_SIZE, GRID_SIZE) / (GRID_SIZE / 10)
    y = torch.linspace(1, GRID_SIZE, GRID_SIZE) / (GRID_SIZE / 10)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    val = U(X, Y)
    val = val.flatten()
    padding = torch.zeros_like(val)
    tmp = torch.stack((X.flatten(), Y.flatten(), padding), dim=1)
    
    data_list = []
    for i in range(GRID_SIZE * GRID_SIZE):
        new_feature = torch.cat((feature, tmp[i].unsqueeze(0)), dim=0)
        data_list.append(Data(x=new_feature, edge_index=edge, y=val[i]))
    
    graph_data = GraphDataSet(data_list)
    loader = DataLoader(graph_data, batch_size=1, shuffle=True)

    return loader

def prepare_testing_data():
    feature, edge = generate_features()
    for i in range(GRID_SIZE * GRID_SIZE):
        edge.append([GRID_SIZE * GRID_SIZE, i])
    edge = torch.tensor(edge, dtype=torch.long).t().contiguous()
    
    x = torch.rand(TEST_NUM, 1) * 10
    y = torch.rand(TEST_NUM, 1) * 10
    val = U(x, y)
    val = val.flatten()
    padding = torch.zeros_like(x)
    tmp = torch.stack((x.flatten(), y.flatten(), padding.flatten()), dim=1)
    
    data_list = []
    for i in range(TEST_NUM):
        new_feature = torch.cat((feature, tmp[i].unsqueeze(0)), dim=0)
        data_list.append(Data(x=new_feature, edge_index=edge, y=val[i]))
    
    graph_data = GraphDataSet(data_list)
    loader = DataLoader(graph_data, batch_size=1, shuffle=True)

    return loader

if __name__ == "__main__":
    prepare_training_data()