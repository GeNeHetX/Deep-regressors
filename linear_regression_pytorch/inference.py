import torch
import torch.nn as nn
import numpy as np

from dataset import MALDI_multisamples
from model import LinearRegression


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
    # --- Configuration ---
    PATH = "data/MALDI_IHC/correlations/"
    PEAKS_PATH = f"{PATH}peaks_standardized_lasso.pkl"
    PIXELS_PATH = f"{PATH}pixels_filtered_lasso.pkl"
    TARGET = 'Density_CD8'
    MODEL_PATH = 'models/linear_regression.pth'

    # --- Load the data ---
    dataset = MALDI_multisamples(peaks=PEAKS_PATH, pixels=PIXELS_PATH, target=TARGET)
    data_tensor = dataset.features  # Example: use the features for prediction
    print(f"Data tensor shape: {data_tensor.shape}")
    print("Data tensor loaded.")

    # --- Load the model ---
    input_dim = dataset.n_features
    output_dim = 1 

    # Load the model
    model = LinearRegression(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # --- Make predictions ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    predictions = predict(model, data_tensor, device)
    print("Predictions made.")
    print(f"Predictions shape: {predictions.shape}")

    # Optionally, save predictions to a file
    np.save('predictions.npy', predictions)
    print("Predictions saved to predictions.npy.")