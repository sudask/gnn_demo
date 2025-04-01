import torch
from torch import nn
from torch_geometric.nn import GCNConv

class MyData:
    def __init__(self, feature, edge_index, vals = None):
        self.feature = feature
        self.edge_index = edge_index
        self.vals = vals

class model19(nn.Module):
    def __init__(self):
        super(model19, self).__init__()
        self.conv1 = GCNConv(1, 3)
        self.conv2 = GCNConv(3, 1)
        self.fc = nn.Linear(19, 1600)

    def forward(self, data):
        feature, edge_index = data.feature, data.edge_index

        feature = self.conv1(feature, edge_index)
        feature = torch.tanh(feature)
        feature = self.conv2(feature, edge_index)
        feature = torch.tanh(feature)
        feature = feature.squeeze()
        feature = self.fc(feature)

        return feature.squeeze()

class model94(nn.Module):
    def __init__(self):
        super(model94, self).__init__()
        self.conv1 = GCNConv(1, 4)
        self.conv2 = GCNConv(4, 1)
        self.fc = nn.Linear(94, 6400)

    def forward(self, data):
        feature, edge_index = data.feature, data.edge_index

        feature = self.conv1(feature, edge_index)
        feature = torch.tanh(feature)
        feature = self.conv2(feature, edge_index)
        feature = torch.tanh(feature)

        feature = feature.squeeze()
        feature = self.fc(feature)

        return feature.squeeze()
    
class model415(nn.Module):
    def __init__(self):
        super(model415, self).__init__()
        self.conv1 = GCNConv(1, 3)
        self.conv2 = GCNConv(3, 1)
        self.fc = nn.Linear(415, 25600)

    def forward(self, data):
        feature, edge_index = data.feature, data.edge_index

        feature = self.conv1(feature, edge_index)
        feature = torch.tanh(feature)
        feature = self.conv2(feature, edge_index)
        feature = torch.tanh(feature)
        feature = feature.squeeze()
        feature = self.fc(feature)

        return feature.squeeze()