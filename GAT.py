import torch
import torch_geometric
from torch_geometric.nn import GATv2Conv


class GAT(torch.nn.Module):
    def __init__(self, num_hidden, heads=3):
        super().__init__()
        # self.linear1=torch.nn.Linear(num_features,num_hidden)
        self.conv1 = GATv2Conv(num_hidden, num_hidden, heads=heads,dropout=0)
        self.conv2 = GATv2Conv(num_hidden, num_hidden, heads=heads,dropout=0)
        self.BN1 = torch.nn.BatchNorm1d(num_hidden)
        self.BN2 = torch.nn.BatchNorm1d(num_hidden)
        # GAT_layers=[self.conv1,torch.nn.BatchNorm1d(num_hidden),torch.nn.ReLU(),
        #             self.conv2,torch.nn.BatchNorm1d(num_hidden),torch.nn.ReLU()]
        # self.GAT=torch.nn.Sequential(*GAT_layers)
        self.linear1 = torch.nn.Linear(num_hidden * heads, num_hidden)
        self.linear2 = torch.nn.Linear(num_hidden * heads*2, num_hidden)

    def forward(self, x, data):
        x_in = self.conv1(x, data.edge_index)
        x = self.linear1(x_in)
        x = self.BN1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, data.edge_index)
        x = self.linear2(torch.cat([x,x_in],dim=1))
        x = self.BN2(x)
        x = torch.nn.functional.relu(x)
        return x
