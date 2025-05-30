import torch_geometric
import torch
from GIN import GIN
from torch_geometric.nn import norm

from torch_geometric.utils import group_argsort, index_to_mask, unbatch, softmax


class actor(torch.nn.Module):
    def __init__(self, num_features, num_hidden, heads=3):
        super().__init__()
        self.linear0 = torch.nn.Linear(num_features, num_hidden)
        self.GIN = GIN(num_hidden * 2)
        self.GIN_embed=GIN(num_hidden)
        self.linear1 = torch.nn.Linear(num_hidden * 4, num_hidden)
        self.linear2 = torch.nn.Linear(num_hidden + num_features, 1)
        self.initial_embed = torch.nn.Parameter(torch.randn(num_hidden))
        self.bn1=norm.BatchNorm(num_hidden,track_running_stats=False)

    def forward(self, data, inf=10**6, prev_node=None,x0=None):
        cities = data.x.size(0) / data.num_graphs  
        batch_index = cities * torch.ones(data.num_graphs)  # batch_index= (num_batch,)
        if x0 is None:
            x0 = self.linear0(data.x)  # size(x0) = (num_nodes_batch, num_hidden)
            x0= self.bn1(x0)
            x0=self.GIN_embed(x0,data)
        if prev_node is not None:
            x0_prev = torch.repeat_interleave(
            x0[prev_node, :].to('cpu'), batch_index.int().to('cpu'), dim=0).to('mps')
            x_in = torch.cat([x0_prev, x0], dim=1)
        else:
            x_in = torch.cat(
                [self.initial_embed.repeat(x0.size(0), 1), x0], dim=1
            )  # size(x_in) = (num_nodes_batch, num_hidden*2)
        enc_out = self.GIN(
            x_in, data
        )  # size(enc_out) = (num_nodes_batch, num_hidden*2)
        batch = data.batch  # size(batch) = (num_nodes_batch,1)
        center_node_embed = enc_out[
            data.center_node_index, :
        ]  # size(center_node_embed) = (num_batch, num_hidden*2)
        
        center_node_embed = torch.repeat_interleave(
            center_node_embed.to('cpu'), batch_index.int().to('cpu'), dim=0
        ).to('mps')  # size(center_node_embed) = (num_nodes_batch, num_hidden*2)
        x = self.linear1(torch.cat([center_node_embed, enc_out], dim=1))
        x_out = torch.nn.functional.relu(x)
        x_lin2 = self.linear2(torch.cat([data.x, x_out], dim=1)).squeeze()
        x = x_lin2 - torch.where(data.mask, 0, inf).squeeze()
        x = softmax(x, index=batch)
        x = torch.stack(unbatch(x, batch))
        if self.training:
            sample = torch.multinomial(x, num_samples=1)
        else:
            _, sample = torch.max(x, dim=1)
            sample = sample.unsqueeze(-1)
        log_action = torch.log(torch.gather(x, index=sample, dim=1)).squeeze()
        return sample.squeeze() + data.graph_id_index, log_action,x0


if __name__ == "__main__":
    from data_generator import data_generator

    loader = data_generator(5, 3, 2)
    for data in loader:
        model = actor(4, 8, 2).to("cpu")
        data.to("cpu")
        print(model(data))
        break

