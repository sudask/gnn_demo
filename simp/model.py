from common import *

from torch_geometric.nn import GCNConv

class SimpleGNN(nn.Module):
    def __init__(self):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)
        
    def forward(self, data):
        x, edge = data.x, data.edge_index
        x = self.conv1(x, edge)
        x = torch.tanh(x)
        x = self.conv2(x, edge)
        return x.flatten()[-1]