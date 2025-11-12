import torch
from torch import nn
from torch_geometric.nn import GCNConv

class MyData:
    def __init__(self, feature, edge_index, grid_coords, vals = None):
        self.feature = feature
        self.edge_index = edge_index
        self.vals = vals
        self.grid_coords = grid_coords
    
class GeneralModel(nn.Module):
    def __init__(self, numObs, numTarget):
        super(GeneralModel, self).__init__()
        self.conv1 = GCNConv(3, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 4)
        self.conv4 = GCNConv(4, 4)
        
        
        self.decoder = nn.Sequential(
            nn.Linear(4 + 2, 5),
            nn.SELU(),
            nn.Linear(5, 5),
            nn.SELU(),
            nn.Linear(5, 1),
        )

    def forward(self, data):
        x, edge_index = data.feature, data.edge_index
        grid_coords = data.grid_coords

        # --- Graph Encoder ---
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))
        h = torch.relu(self.conv3(h, edge_index))
        h = torch.relu(self.conv4(h, edge_index))
        
        # --- Decoder ---
        # 取出目标节点
        h_grid = h[-grid_coords.shape[0]:]

        decoder_input = torch.cat([h_grid, grid_coords], dim=-1)

        out = self.decoder(decoder_input)     # (num_target, 1)
        return out.squeeze(-1)