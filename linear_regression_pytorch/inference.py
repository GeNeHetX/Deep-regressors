import torch
import torch.nn as nn
import numpy as np


def predict(model: nn.Module,
            data_tensor: torch.Tensor,
            device: torch.device) -> np.ndarray:
    """
    Makes predictions using the trained model.

    Args:
        model (nn.Module): The trained model.
        data_tensor (torch.Tensor): Input data tensor for prediction
                                   (shape: [n_samples, n_features]).
        device (torch.device): The device the model is on ('cpu' or 'cuda').

    Returns:
        np.ndarray: Predictions as a NumPy array.
    """
    model.to(device)  # Ensure model is on the correct device
    model.eval()      # Set the model to evaluation mode (disables dropout, batch norm updates etc.)

    with torch.no_grad(): # Disable gradient calculation for inference
        data_tensor = data_tensor.to(device)
        predictions = model(data_tensor)

    # Move predictions to CPU (if they were on GPU) and convert to NumPy array
    return predictions.cpu().numpy()

# Example usage (optional, part of main.py usually)
if __name__ == '__main__':
    # This part is typically run from main.py
    print("This script contains the inference function.")
    print("To run inference, execute main.py after training.")
    # You could add placeholder code here for direct testing if needed,
    # using a dummy model and data.