import torch
import torch.nn as nn

def mlp_architecture(input_dim: int, output_dim: int=1, hidden_dim: int=256, num_hidden_layers: int=2, architecture_factor: float = 1, dropout: float=0.0) -> list:
    """
    Build a list of layers for an MLP with different architectures.

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output values.
        hidden_dim (int): Base hidden dimension.
        num_hidden_layers (int): Number of hidden layers.
        dropout (float): Dropout probability.
        architecture_factor (float): Multiplicative factor for hidden layer sizes (1 = Uniform, < 1 = Funnel, > 1 = Expanding).
    Returns:
        list: List of nn.Module layers.
    """
    layers = []  # Initialize an empty list to hold the layers
    previous_dim = input_dim  # Start with the input dimension
    current_dim = hidden_dim  # Set the first hidden layer dimension

    for i in range(num_hidden_layers):
        layers.append(nn.Linear(previous_dim, int(current_dim)))  # Add a linear layer
        if i < num_hidden_layers - 1:  # No activation after the last layer
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        previous_dim = int(current_dim)  # Update previous dimension for the next layer
        current_dim = int(current_dim * architecture_factor)  # Adjust the current dimension based on the architecture factor
    layers.append(nn.Linear(previous_dim, output_dim))
    
    return layers

class MLPRegression(nn.Module):
    """
    Flexible Multi-Layer Perceptron (MLP) for regression using built-in PyTorch modules.
    """
    def __init__(self, input_dim: int, output_dim: int=1, hidden_dim: int=256, num_hidden_layers: int=2, architecture_factor: float = 1, dropout: float=0.0):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output values (usually 1 for regression).
            hidden_dim (int): Number of units in each hidden layer.
            num_hidden_layers (int): Number of hidden layers.
            dropout (float): Dropout probability (0.0 means no dropout).
            architecture_factor (float): Multiplicative factor for hidden layer sizes (1 = Uniform, < 1 = Funnel, > 1 = Expanding).
        """
        super(MLPRegression, self).__init__()
        layers = mlp_architecture(input_dim, output_dim, hidden_dim, num_hidden_layers, architecture_factor, dropout)
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
    model = MLPRegression(input_dim=1, output_dim=1, hidden_dim=512, num_hidden_layers=3, architecture_factor=0.5, dropout=0.2)
    print(model)