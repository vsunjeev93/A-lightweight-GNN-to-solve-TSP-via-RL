# GNN-based Reinforcement Learning for TSP

A Graph Isomorphism Network (GIN) based reinforcement learning approach to solve the Traveling Salesman Problem (TSP) using an actor-critic architecture.

## Architecture
- **Graph Structure**: Cities as nodes with features (coordinates, distance and angle from central node) connected to a central node
- **Actor-Critic Networks**: Both use GIN layers with skip connections
- **GIN Layer**: Two-layer MLPs with feature concatenation and ReLU activations

### Feature Vector Structure
Each node in the graph contains a 4-dimensional feature vector:
- For city nodes: `[x1, y1, d1, a1]`
  - `x1, y1`: City coordinates (normalized to [0,1])
  - `d1`: Distance from the city to the center node
  - `a1`: Angle between the city and center node (in radians)
- For center node: `[x_c, y_c, 0, 0]`
  - `x_c, y_c`: Mean coordinates of all cities
  - Last two dimensions are padded with zeros

![image](https://github.com/user-attachments/assets/a468edf5-5e60-4cdc-b5b7-db988108eb10)


### State Transitions
The environment graph data stucture updates after each city selection through two mechanisms:
1. **Edge Updates**: When a city is selected
   - New bidirectional edges are added between:
     - The previously selected city (if any)
     - The currently selected city
2. **Masking**: Tracks visited cities
   - Each city starts as unmasked (available)
   - After a city is selected, it's masked (marked as visited)
   - The mask ensures each city is visited exactly once

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

## Requirements
- PyTorch
- PyTorch Geometric
- NumPy
