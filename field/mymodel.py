import torch
from torch import nn
from torch_geometric.nn import GCNConv

class MyData:
    def __init__(self, feature, edge_index, vals = None):
        self.feature = feature
        self.edge_index = edge_index
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