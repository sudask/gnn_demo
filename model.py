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