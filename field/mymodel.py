import torch
from torch import nn
from torch_geometric.nn import GCNConv

class MyData:
    def __init__(self, feature, edge_index, grid, vals = None):
        self.feature = feature
        self.edge_index = edge_index
        self.grid = grid
        self.vals = vals
    
class GeneralModel(nn.Module):
    def __init__(self, numObs, numTarget):
        super(GeneralModel, self).__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 16)
        
        self.trunk = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.bias = nn.Parameter(torch.randn(1) * 0.01)

    def forward(self, data):
        feature, edge_index, grid = data.feature, data.edge_index, data.grid
        
        h1 = torch.relu(self.conv1(feature, edge_index))
        h2 = torch.relu(self.conv2(h1, edge_index))
        b_raw = self.conv3(h2, edge_index)

        b = b_raw.mean(dim=0)   # shape (16,)
        b = nn.LayerNorm(b.shape[-1])(b)
        t = self.trunk(grid)

        prod = t * b.unsqueeze(0)   # shape (num_target, 16)
        output = torch.sum(prod, dim=-1) + self.bias
        
        # output = torch.sum(b * t, dim=-1) + self.bias
        
        # output = torch.matmul(t, b.T).sum(dim=-1) + self.bias
        
        # print("final output mean/std:", output.mean().item(), output.std().item())
        
        return output