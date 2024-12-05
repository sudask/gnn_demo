from config import *
from torch_geometric.nn import GCNConv

class SimpleGNN(nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(2, 1)
        self.fc = nn.Linear(GRID_SIZE**2, 1)
        
    def forward(self, data):
        feature, edge = data.x, data.edge_index
        feature = self.conv1(feature, edge)
        feature = torch.relu(feature)
        feature = feature.squeeze()
        
        x = self.fc(feature)

        return x.squeeze()
    
class CompleteGNN(nn.Module):
    def __init__(self):
        super(CompleteGNN, self).__init__()
        self.conv1 = GCNConv(2, 1)
        self.conv2 = GCNConv(2, 1)

        self.fc1 = nn.Linear(2500, 1)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, data):
        feature1, feature2, edge_index1, edge_index2 = data.feature1, data.feature2, data.edge_index1, data.edge_index2

        feature1 = self.conv1(feature1, edge_index1)
        feature2 = self.conv2(feature2, edge_index2)

        feature1 = torch.relu(feature1)
        feature2 = torch.relu(feature2)

        feature1 = feature1.squeeze()
        feature2 = feature2.squeeze()

        x1 = self.fc1(feature1)
        x2 = self.fc2(feature2)

        x = (x1 + x2) * 0.5

        return x.squeeze()