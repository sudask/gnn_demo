from config import *
from torch_geometric.nn import GCNConv
    
class CompleteGNN(nn.Module):
    def __init__(self):
        super(CompleteGNN, self).__init__()
        self.conv11 = GCNConv(2, 8)
        self.conv21 = GCNConv(8, 1)
        self.conv12 = GCNConv(2, 8)
        self.conv22 = GCNConv(8, 1)
        self.fc1 = nn.Linear(2500, 1)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, data):
        feature1, edge_index1, feature2, edge_index2 = data.feature1, data.edge_index1, data.feature2, data.edge_index2

        feature1 = self.conv11(feature1, edge_index1)
        feature1 = torch.tanh(feature1)
        feature1 = self.conv21(feature1, edge_index1)
        feature1 = torch.tanh(feature1)
        feature1 = feature1.squeeze()
        x1 = self.fc1(feature1)

        feature2 = self.conv12(feature2, edge_index2)
        feature2 = torch.tanh(feature2)
        feature2 = self.conv22(feature2, edge_index2)
        feature2 = torch.tanh(feature2)
        feature2 = feature2.squeeze()
        x2 = self.fc2(feature2)

        if TWO_GRID:
            return x1.squeeze() + x2.squeeze()
        else:
            return x1.squeeze()