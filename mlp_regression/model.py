import torch
import torch.nn as nn

class MLPRegression(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) for regression using built-in PyTorch modules.
    """
    def __init__(self, input_dim: int, output_dim: int=1, hidden_dim: int=64, num_hidden_layers: int=2, dropout: float=0.0):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output values (usually 1 for regression).
            hidden_dim (int): Number of units in each hidden layer.
            num_hidden_layers (int): Number of hidden layers.
            dropout (float): Dropout probability (0.0 means no dropout).
        """
        super(MLPRegression, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        return self.mlp(x)

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # Example: 2 input features, 1 output value
    model = MLPRegression(input_dim=2, output_dim=1)
    print(model)
    # Print parameter names and shapes
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")