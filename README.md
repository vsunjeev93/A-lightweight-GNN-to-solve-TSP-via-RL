# GNN-based Reinforcement Learning for TSP

A Graph Isomorphism Network (GIN) based reinforcement learning approach to solve the Traveling Salesman Problem (TSP) using an actor-critic architecture.

## Architecture
- **Graph Structure**: Cities as nodes with features (coordinates, distance and angle from central node) connected to a central node
- **Actor-Critic Networks**: Both use GIN layers with skip connections and batch normalization
- **GIN Layer**: Two-layer MLPs with feature concatenation and ReLU activations

![image](https://github.com/user-attachments/assets/98b712dd-7898-45b3-b227-366ff6740ad7)

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
