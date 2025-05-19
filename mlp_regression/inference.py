import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from dataset import MALDI_multisamples
from model import MLPRegression


def predict(model: nn.Module,
            data_tensor: torch.Tensor,
            device: torch.device,
            batch_size: int = 512) -> np.ndarray:
    """
    Makes predictions using the trained model in batches.

    Args:
        model (nn.Module): The trained model.
        data_tensor (torch.Tensor): Input data tensor for prediction
                                   (shape: [n_samples, n_features]).
        device (torch.device): The device the model is on ('cpu' or 'cuda').
        batch_size (int): Batch size for prediction.

    Returns:
        np.ndarray: Predictions as a NumPy array.
    """
    model.to(device)  # Ensure model is on the correct device
    model.eval()      # Set the model to evaluation mode (disables dropout, batch norm updates etc.)

    predictions = []
    n_samples = data_tensor.shape[0]
    with torch.no_grad(): # Disable gradient calculation for inference
        for start in tqdm(range(0, n_samples, batch_size), desc="Predicting", unit="batch"):
            end = min(start + batch_size, n_samples)
            batch = data_tensor[start:end].to(device)
            batch_preds = model(batch)
            predictions.append(batch_preds.cpu())
    predictions = torch.cat(predictions, dim=0)
    return predictions.numpy()

# Example usage (optional, part of main.py usually)
if __name__ == '__main__':
    # --- Configuration ---
    PATH = "data/MALDI_IHC/correlations/"
    PEAKS_PATH = f"{PATH}peaks_standardized.pkl"
    PIXELS_PATH = f"{PATH}pixels_filtered.pkl"
    TARGET = 'Density_CD8'
    MODEL_PATH = 'models/MLP_regression.pth'
    BATCH_SIZE = 10**5

    # --- Load the data ---
    dataset = MALDI_multisamples(peaks=PEAKS_PATH, pixels=PIXELS_PATH, target=TARGET)
    data_tensor = dataset.features  # Example: use the features for prediction
    print(f"Data tensor shape: {data_tensor.shape}")
    print("Data tensor loaded.")

    # --- Load the model ---
    input_dim = dataset.n_features
    output_dim = 1 

    # Load the model
    model = MLPRegression(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # --- Make predictions ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    predictions = predict(model, data_tensor, device, batch_size=BATCH_SIZE)
    print("Predictions are done.")
    print(f"Predictions shape: {predictions.shape}")

    # Optionally, save predictions to a file
    np.save('predictions_mlp.npy', predictions)
    print("Predictions saved to predictions_mlp.npy.")