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
        self.conv1 = GCNConv(3, 2)
        self.conv2 = GCNConv(2, 1)
        self.fc1 = nn.Linear(94, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc = nn.Linear(128, 6400)

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
    
class model29(nn.Module):
    def __init__(self):
        super(model29, self).__init__()
        self.conv1 = GCNConv(3, 2)
        self.conv2 = GCNConv(2, 1)
        self.fc1 = nn.Linear(29, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc = nn.Linear(128, 1296)

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
    
class model28(nn.Module):
    def __init__(self):
        super(model28, self).__init__()
        self.conv1 = GCNConv(3, 2)
        self.conv2 = GCNConv(2, 1)
        self.fc1 = nn.Linear(28, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc = nn.Linear(512, 1800)

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