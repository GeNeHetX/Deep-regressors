import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import joblib

from dataset import TableDataset
from model import MLPRegression
from utils import get_target_transform, get_inverse_transform, perform_dim_reduction


def predict(model: nn.Module,
            data_tensor: torch.Tensor,
            device: torch.device,
            batch_size: int = 512) -> np.ndarray:
    """
    Makes predictions using the trained model in batches.

    Args:
        model (nn.Module): The trained model.
        data_tensor (torch.Tensor): Input data tensor for prediction (shape: [n_samples, n_features]).
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
    with open("mlp_regression/config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)  # Load model configuration from YAML file

    PATH = config.get('path_to_data_inference')
    PEAKS_PATH = config.get('peaks_path_inference')
    PIXELS_PATH = config.get('pixels_path_inference')
    TARGET = config.get('target_inference')
    EXCLUDED_SLIDES = config.get('excluded_slides')  # Default to empty list if not provided
    BATCH_SIZE = config.get('batch_size_inference')
    HUBER_DELTA = config.get('huber_delta')  # Huber loss delta value

    # MLP Hyperparameters
    HIDDEN_DIM = config.get('hidden_dim')  # Number of neurons in hidden layers
    NUM_HIDDEN_LAYERS = config.get('num_hidden_layers')  # Number of hidden layers
    ARCHITECTURE_FACTOR = config.get('architecture_factor') # Factor to adjust hidden layer sizes (1 = Uniform, < 1 = Funnel, > 1 = Expanding)
    DROPOUT = config.get('dropout')  # Dropout rate
    TARGET_TRANSFORM = config.get('target_transform')  # e.g., 'sqrt', 'log', or 'none'

    # Dimensionality Reduction Hyperparameters
    REDUCTION_METHOD = config.get('reduction_method')
    REDUCTION_N_COMPONENT = config.get('reduction_n_component')


    MODEL_SUFFIX = f"{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}_{ARCHITECTURE_FACTOR}_{REDUCTION_N_COMPONENT}_{REDUCTION_METHOD}"
    MODEL_PATH = f'results/models/MLP_regression_{MODEL_SUFFIX}.pth'
    MODEL_BASE_PATH = f"results/models/{REDUCTION_N_COMPONENT}"
    PREDICTIONS_PATH = f'results/predictions/predictions_mlp_{MODEL_SUFFIX}.npy'

    # Define target transformation functions
    target_transform = get_target_transform(TARGET_TRANSFORM)
    inverse_transform = get_inverse_transform(TARGET_TRANSFORM)

    # Load Data
    print("Loading data...")
    peaks = pd.read_pickle(PEAKS_PATH)
    pixels = pd.read_pickle(PIXELS_PATH)

    # Clean the data by dropping excluded slides
    if EXCLUDED_SLIDES:
        print(f"Dropping excluded slides...")
        mask = ~pixels['run'].isin(EXCLUDED_SLIDES)
        peaks = peaks[mask].reset_index(drop=True)
        pixels = pixels[mask].reset_index(drop=True)

    # Perform dimensionality reduction
    if REDUCTION_N_COMPONENT is not None:
        print(f"Applying {REDUCTION_METHOD.upper()} with n_components={REDUCTION_N_COMPONENT} to features...")
        peaks = perform_dim_reduction(
            features=peaks.values,
            n_components=REDUCTION_N_COMPONENT,
            model_base_path=MODEL_BASE_PATH,
            method=REDUCTION_METHOD
        )
    else:
        print("Skipping dimensionality reduction.")
        peaks = peaks.values

    # Pass cleaned arrays/DataFrames to the dataset
    print("Creating dataset...")
    dataset = TableDataset(
        features=peaks,
        target=pixels[TARGET].values,
        target_transform=target_transform
    )

    # --- Load the model ---
    input_dim = dataset.n_features
    output_dim = 1 

    # Load the model
    print("Loading the model...")
    model = MLPRegression(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        architecture_factor=ARCHITECTURE_FACTOR,
        dropout=DROPOUT
    )
    print(model)
    
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # --- Make predictions ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Making predictions...")
    predictions = predict(model, dataset.features, device, batch_size=BATCH_SIZE)
    print("Predictions are done.")
    print(f"Predictions shape: {predictions.shape}")

    # Invert the transformation before saving
    predictions = inverse_transform(predictions)

    # Optionally, save predictions to a file
    np.save(PREDICTIONS_PATH, predictions)
    print(f"Predictions saved to {PREDICTIONS_PATH}.")