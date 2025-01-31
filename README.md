# GNN-based Reinforcement Learning for TSP

A lightweight Graph Neural Network (GNN) based reinforcement learning approach to solve the Traveling Salesman Problem (TSP) using an actor-critic architecture with Graph Isomorphism Networks (GIN).

## Architecture
- **Graph Structure**: Cities as nodes with features (coordinates, distance from center, angle) connected to a central node
- **Actor-Critic Networks**: Both use GIN layers with skip connections and batch normalization
- **GIN Layer**: Two-layer MLPs with feature concatenation and ReLU activations
graph TD
    subgraph Graph Structure
        A["Center Node<br>[x_c, y_c, 0, 0]"] --> B["City 1<br>[x1, y1, d1, a1]"]
        A --> C["City 2<br>[x2, y2, d2, a2]"]
        A --> D["City 3<br>[x3, y3, d3, a3]"]
        A --> E["City N<br>[xn, yn, dn, an]"]
    end

    subgraph GNN Architecture
        G[Input Features] --> H[Linear Layer]
        H --> I[GIN Layer 1]
        H --Skip--> J[Concat 1]
        I --> J
        J --> K[GIN Layer 2]
        H --Skip--> L[Concat 2]
        K --> L
        L --> M[Linear Layer]
        M --> N[Output]
    end

    subgraph GIN Layer
        X[Input] --> M1[MLP1 + GINConv1]
        X --Skip--> C1[Concat]
        M1 --> C1
        C1 --> M2[MLP2 + GINConv2]
        M2 --> F[Linear + ReLU]
    end

## Usage

Training:
```bash
python train.py --city 20 --batch_size 20 --instances 1000 --epoch 20 --embed 128 --steps_per_epoch 100
```

Testing:
```bash
python test.py --city 20 --embed 128
```

### Key Parameters
- `--city`: Number of cities in TSP instances
- `--batch_size`: Batch size for training
- `--embed`: Dimension of embedding layers (default: 128)
- `--LR`: Learning rate (default: 0.0001)
- `--decay`: Learning rate decay (default: 0.9)

## Implementation Details
- PyTorch Geometric for GNN operations
- State transitions via edge updates and node masking
- Kaiming initialization and gradient clipping (max norm: 1.0)
- Learning rate decay for optimization

## Requirements
- PyTorch
- PyTorch Geometric
- NumPy