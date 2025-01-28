import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import index_to_mask


def generate_graph_from_data(n, points):

    center = torch.mean(points, dim=0)
    distance = torch.linalg.norm(points - center, dim=1)
    angle_center = torch.atan2(points[:, 1] - center[1], points[:, 0] - center[0])
    features = []
    edges = []
    angle_center_list = []
    # print(points)
    k = 3
    for i in range(n):
        distances = torch.linalg.norm(points - points[i], dim=1)
        # print(distances)
        knn_distances, knn_indices = torch.topk(distances, k=k + 1, largest=False)
        # Exclude self-distance
        knn_features = knn_distances[1:]
        knn_indices = knn_indices[1:]
        # for j in range(k):
        #     edges.append(torch.tensor([i, knn_indices[j]]))
        #     edges.append(torch.tensor([knn_indices[j], i]))
        # print(knn_features)
        angle_center_list.append(angle_center[i])
        # features.append(
        #     torch.tensor([points[i, 0], points[i, 1], distance[i], angle_center[i]])
        # )
        # print(points[i].size(),distance[i],angle_center[i].size(),'size')
        features.append(
            torch.cat(
                [
                    points[i],
                    distance[i].unsqueeze(0),
                    angle_center[i].unsqueeze(0),
                ]
            )
        )
        # print(features)
        # assert 1==2
        edges.append(torch.tensor([n, i]))
        edges.append(torch.tensor([n, i]))
    features.append(torch.tensor([center[0], center[1], 0, 0]))
    # print(max(angle_center_list),min(angle_center_list),'min max')
    features = torch.stack(features)
    edge_index = torch.stack(edges)
    mask = index_to_mask(torch.tensor(range(n)), size=n + 1)
    graph = Data(
        x=features,
        edge_index=edge_index.t().contiguous(),
        center_node_index=torch.tensor([n]),
        mask=mask,
        graph_id_index=torch.tensor([0]),
    )
    return graph


def data_generator(n, instances=10000, batch_size=12):
    graphs = []
    np.random.seed(42)
    for instance in range(instances):
        # print(n)
        points = torch.rand((n, 2))
        graph = generate_graph_from_data(n, points)
        graphs.append(graph)
    loader = DataLoader(graphs, batch_size=batch_size)
    return loader


# loader = data_generator(5, 3, 1)
# print(loader)
