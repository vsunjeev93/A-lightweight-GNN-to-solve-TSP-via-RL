import torch
import torch_geometric
from torch_geometric.nn import GINConv, norm


class GIN(torch.nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.linear1=torch.nn.Linear(num_hidden,num_hidden)
        self.linear2=torch.nn.Linear(num_hidden,num_hidden)
        self.linear3=torch.nn.Linear(num_hidden+num_hidden,num_hidden)
        self.linear4=torch.nn.Linear(num_hidden,num_hidden)
        self.MLP1=torch.nn.Sequential(self.linear1,torch.nn.ReLU(),
                                      self.linear2,torch.nn.ReLU())
        self.MLP2=torch.nn.Sequential(self.linear3,torch.nn.ReLU(),
                                      self.linear4,torch.nn.ReLU())
        self.conv1 = GINConv(self.MLP1,)
        self.conv2 = GINConv(self.MLP2)
        self.linear5=torch.nn.Linear(num_hidden,num_hidden)

    def forward(self, x, data):
        x_in = self.conv1(x, data.edge_index)
        x_in=torch.cat([x,x_in],dim=1)
        x_in=self.conv2(x_in,data.edge_index)
        x = self.linear5(x_in)
        x = torch.nn.functional.relu(x)
       
        return x
