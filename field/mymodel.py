import torch
from torch import nn
from torch_geometric.nn import GCNConv

class MyData:
    def __init__(self, feature, edge_index, vals = None):
        self.feature = feature
        self.edge_index = edge_index
        self.vals = vals

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = GCNConv(1, 3)
        self.conv2 = GCNConv(3, 1)
        self.fc = nn.Linear(49, 2500)

    def forward(self, data):
        feature, edge_index = data.feature, data.edge_index

        feature = self.conv1(feature, edge_index)
        feature = torch.tanh(feature)
        feature = self.conv2(feature, edge_index)
        feature = torch.tanh(feature)
        feature = feature.squeeze()
        feature = self.fc(feature)

        return feature.squeeze()