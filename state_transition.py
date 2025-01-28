import torch


def state_transition(data, current_node, prev_node):
    if prev_node is not None:
        new_edges = torch.stack(
            [
                torch.cat([prev_node, current_node]),
                torch.cat([current_node, prev_node]),
            ],
            dim=0,
        )
        data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    data.mask[current_node.squeeze()] = False
    # data.x=data.x.clone()
    # data.x[current_node, -1] = 1
    return data
