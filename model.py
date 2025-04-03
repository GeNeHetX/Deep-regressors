import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    """
    Simple Linear Regression model using a single linear layer.
    """
    def __init__(self, input_dim: int, output_dim: int=1):
        """
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output values (usually 1 for regression).
        """
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        return self.linear(x)

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # Example: 2 input features, 1 output value
    model = LinearRegression(input_dim=2, output_dim=1)
    print(model)
    # Print parameter names and shapes
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}, Shape: {param.shape}")