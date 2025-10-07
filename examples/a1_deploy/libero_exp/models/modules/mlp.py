from torch import nn


class DynamicMLP(nn.Module):
    def __init__(self, layer_dims):
        """
        Initialize a dynamically configurable MLP.

        Args:
            layer_dims (list of int): List of layer dimensions, where
                                      len(layer_dims) determines the number of layers.
                                      For example, [10, 64, 32, 16] creates a 3-layer MLP:
                                      - Input: 10
                                      - Hidden1: 64
                                      - Hidden2: 32
                                      - Output: 16
        """
        super(DynamicMLP, self).__init__()

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:  # No activation or normalization on the output layer
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    