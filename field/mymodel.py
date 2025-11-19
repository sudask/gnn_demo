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
        self.conv3 = GCNConv(4, 4)
        
        self.trunk = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        feature, edge_index, grid = data.feature, data.edge_index, data.grid
        
        h = torch.relu(self.conv1(feature, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        b = torch.relu(self.conv3(h, edge_index))
        
        b = b.mean(dim=0)
        
        t = self.trunk(grid)
        
        output = torch.sum(b * t, dim=-1) + self.bias
        
        # branch_features = torch.relu(self.conv2(h, edge_index))
        # branch_output = torch.mean(branch_features, dim=0, keepdim=True)
        
        # # 添加调试输出
        # print("Branch output range:", branch_output.min().item(), branch_output.max().item())
        # print("Branch output mean:", branch_output.mean().item())
        
        return output
    
class DeepONetGNN(nn.Module):
    def __init__(self, branch_in_dim: int = 3, branch_channels: int = 32, p: int = 64, trunk_hidden: int = 64, trunk_layers: int = 3, dropout: float = 0.0):
        """
        Args:
        branch_in_dim: input feature dim for nodes (default 3: x,y,temp)
        branch_channels: hidden channels inside GCN
        p: latent dimension (must match trunk output dim)
        trunk_hidden: width of trunk MLP hidden layers
        trunk_layers: number of trunk hidden layers (>=1)
        dropout: trunk dropout
        """
        super().__init__()
        # Branch (GCN)
        self.conv1 = GCNConv(branch_in_dim, branch_channels)
        self.conv2 = GCNConv(branch_channels, branch_channels)
        # final conv produces p-dim embedding per node; do not apply ReLU here
        self.conv3 = GCNConv(branch_channels, p)


        # Optional small projection / normalization on branch aggregated vector
        self.branch_ln = nn.LayerNorm(p)


        # Trunk (MLP) maps coord (2d) -> p
        trunk_layers_list = []
        in_dim = 2
        for i in range(trunk_layers):
            trunk_layers_list.append(nn.Linear(in_dim, trunk_hidden))
            trunk_layers_list.append(nn.ReLU())
            if dropout > 0:
                trunk_layers_list.append(nn.Dropout(dropout))
            in_dim = trunk_hidden
        # final mapping to p dims
        trunk_layers_list.append(nn.Linear(in_dim, p))
        self.trunk = nn.Sequential(*trunk_layers_list)


        # small bias term (scalar) similar to your original model
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, data: MyData) -> torch.Tensor:
        """
        Args:
        data: MyData with feature [numObs, 3], edge_index [2, E], grid [numTarget, 2]
        Returns:
        temps: Tensor [numTarget] predicted temperatures at each grid coordinate
        """
        x, edge_index, grid = data.feature, data.edge_index, data.grid


        # -- Branch (GCN) --
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        # final conv: do NOT relu so b can be signed
        h = self.conv3(h, edge_index) # [numObs, p]


        # aggregate across observation nodes -> branch vector b [p]
        # here we use mean; you may switch to sum or an attention aggregator
        b = h.mean(dim=0)
        b = self.branch_ln(b)


        # -- Trunk (MLP) --
        # grid: [numTarget, 2] -> t: [numTarget, p]
        t = self.trunk(grid)


        # -- Combine: dot product along latent dim -> [numTarget]
        # ensure shapes: b [p] -> expand to [numTarget, p]
        out = torch.sum(t * b.unsqueeze(0), dim=1) + self.bias
        return out