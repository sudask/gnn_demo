from config import *

from torch_geometric.nn import GCNConv

class SimpleGNN(nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(2, 5)
        self.conv2 = GCNConv(5, 1)
        self.fc = nn.Linear(10, 1)
        
    def forward(self, data):
        feature, edge = data.x, data.edge_index
        feature = self.conv1(feature, edge)
        feature = torch.relu(feature)
        feature = self.conv2(feature, edge)
        feature = torch.relu(feature)
        # feature = torch.squeeze(feature)
        # x = self.fc(feature)
        
        feature = torch.squeeze(feature)
        _, indices = torch.topk(torch.abs(feature), k=10)
        new_feature = feature[indices]
        x = self.fc(new_feature)

        return x.squeeze()