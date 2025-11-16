import torch
from torch import nn
from torch_geometric.nn import GCNConv

class MyData:
    def __init__(self, feature, edge_index, grid, vals = None):
        self.feature = feature
        self.edge_index = edge_index
        self.grid = grid
        self.vals = vals
    
class GeneralModel(nn.Module):
    def __init__(self, numObs, numTarget):
        super(GeneralModel, self).__init__()
        self.conv1 = GCNConv(3, 2)
        self.conv2 = GCNConv(2, 1)
        self.fc1 = nn.Linear(numObs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc = nn.Linear(128, numTarget)

    def forward(self, data):
        feature, edge_index = data.feature, data.edge_index

        feature = self.conv1(feature, edge_index)
        feature = torch.relu(feature)
        feature = self.conv2(feature, edge_index)
        feature = torch.relu(feature)

        feature = feature.squeeze()

        feature = self.fc1(feature)
        feature = torch.relu(feature)
        feature = self.fc2(feature)
        feature = torch.relu(feature)
        feature = self.fc(feature)

        return feature.squeeze()
    
class DeepOnet(nn.Module):
    def __init__(self, numObs, numTarget):
        super(DeepOnet, self).__init__()
        self.conv1 = GCNConv(3, 4)
        self.conv2 = GCNConv(4, 1)
        
        self.trunk = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 28)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        feature, edge_index, grid = data.feature, data.edge_index, data.grid
        
        h = torch.relu(self.conv1(feature, edge_index))
        b = torch.relu(self.conv2(h, edge_index))
        
        t = self.trunk(grid)
        
        output = torch.sum(b.squeeze(-1) * t, dim=1) + self.bias
        
        return output
        
        
        
        