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
        self.conv1 = GCNConv(3, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 1)
        
        self.trunk = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 28)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        feature, edge_index, grid = data.feature, data.edge_index, data.grid
        
        h = torch.relu(self.conv1(feature, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        b = torch.relu(self.conv3(h, edge_index))
        
        t = self.trunk(grid)
        
        output = torch.sum(b.squeeze(-1) * t, dim=1) + self.bias
        
        branch_features = torch.relu(self.conv2(h, edge_index))
        branch_output = torch.mean(branch_features, dim=0, keepdim=True)
        
        # 添加调试输出
        print("Branch output range:", branch_output.min().item(), branch_output.max().item())
        print("Branch output mean:", branch_output.mean().item())
        
        return output