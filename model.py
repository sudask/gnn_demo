from config import *
from torch_geometric.nn import GCNConv
    
class CompleteGNN(nn.Module):
    def __init__(self):
        super(CompleteGNN, self).__init__()
        self.conv1 = GCNConv(2, 4)
        self.conv2 = GCNConv(4, 8)
        self.conv3 = GCNConv(8, 4)
        self.conv4 = GCNConv(4, 1)
        if TWO_GRID:
            self.fc = nn.Linear(2600, 1)
        else:
            self.fc = nn.Linear(2500, 1)

    def forward(self, data):
        feature, edge_index = data.feature, data.edge_index

        feature = self.conv1(feature, edge_index)
        feature = torch.tanh(feature)
        feature = self.conv2(feature, edge_index)
        feature = torch.tanh(feature)
        feature = self.conv3(feature, edge_index)
        feature = torch.tanh(feature)
        feature = self.conv4(feature, edge_index)
        feature = torch.tanh(feature)
        feature = feature.squeeze()
        feature = self.fc(feature)

        return feature.squeeze()