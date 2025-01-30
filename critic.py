import torch_geometric
import torch
from GAT import GAT
from GIN import GIN
from torch_geometric.utils import group_argsort, index_to_mask, unbatch, softmax


class critic(torch.nn.Module):
    def __init__(self, num_features, num_hidden, heads=3):
        super().__init__()
        self.linear0 = torch.nn.Linear(num_features, num_hidden)
        self.GIN = GIN(num_hidden * 2)
        self.linear1 = torch.nn.Linear(num_hidden * 2, num_hidden)
        self.linear2 = torch.nn.Linear(num_hidden, 1)
        self.initial_embed = torch.nn.Parameter(torch.randn(num_hidden))

    def forward(self, data, inf=10**8):
        x0 = self.linear0(data.x)  # size(x0) = (num_nodes_batch, num_hidden)
        x_in = torch.cat(
            [self.initial_embed.repeat(x0.size(0), 1), x0], dim=1
        )  # size(x_in) = (num_nodes_batch, num_hidden*2)
        enc_out = self.GIN(
            x_in, data
        )  # size(enc_out) = (num_nodes_batch, num_hidden*2)
        center_node_embed = enc_out[
            data.center_node_index, :
        ]  # size(center_node_embed) = (num_batch, num_hidden*2)
        x = self.linear1(center_node_embed)  # size(x) = (num_batch, num_hidden)
        x=torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    from data_generator import data_generator

    loader = data_generator(5, 3, 2)
    for data in loader:
        model = critic(4, 8, 2).to("cpu")
        data.to("cpu")
        print(model(data))
        break
