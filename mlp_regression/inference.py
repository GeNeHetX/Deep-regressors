import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import joblib
import gc
import os

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from dataset import TableDataset
from model import MLPRegression
from utils import get_target_transform, get_inverse_transform, perform_dim_reduction

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

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

    # Optimization Hyperparameters
    HUBER_DELTA = config.get('huber_delta')
    LEARNING_RATE = config.get('learning_rate')
    MAX_LR = config.get('max_lr')
    WEIGHT_DECAY = config.get('weight_decay')

    # MLP Hyperparameters
    HIDDEN_DIM = config.get('hidden_dim')  # Number of neurons in hidden layers
    NUM_HIDDEN_LAYERS = config.get('num_hidden_layers')  # Number of hidden layers
    ARCHITECTURE_FACTOR = config.get('architecture_factor') # Factor to adjust hidden layer sizes (1 = Uniform, < 1 = Funnel, > 1 = Expanding)
    DROPOUT = config.get('dropout')  # Dropout rate

    # Target and Features Transformations
    TARGET_TRANSFORM = config.get('target_transform')
    FEATURES_TRANSFORM = config.get('features_transform')

    # Dimensionality Reduction Hyperparameters
    REDUCTION_METHOD = config.get('reduction_method')
    REDUCTION_N_COMPONENT = config.get('reduction_n_component')
    ICA = config.get('ica', False)  # Check if ICA is enabled

    # Define model suffix and paths
    MODEL_SUFFIX = f"{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}_{ARCHITECTURE_FACTOR}_{REDUCTION_N_COMPONENT}_{REDUCTION_METHOD}{'_ica' if ICA else ''}_{HUBER_DELTA}_{LEARNING_RATE}_{MAX_LR}_{WEIGHT_DECAY}"
    MODEL_PATH = f'results/models/MLP_regression_{MODEL_SUFFIX}.pth'
    MODEL_BASE_PATH = f"results/models/{REDUCTION_N_COMPONENT}"
    PREDICTIONS_PATH = f'results/predictions/predictions_mlp_{MODEL_SUFFIX}.npy'

    # Define target transformation functions
    target_transform = get_target_transform(TARGET_TRANSFORM)
    inverse_transform = get_inverse_transform(TARGET_TRANSFORM)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # Extract unique slides and their count
    slides = pixels['run'].unique()
    n_slides = len(slides)

    # Drop the peaks that are in the trypsin peptide masses with tolerance 0.2
    with open("trypsin_peaks.yaml", "r") as f:
        trypsin_peaks = yaml.safe_load(f)

    for col in peaks.columns:
        if np.min(np.abs(float(col) - np.array(trypsin_peaks))) < 0.2:
            print(f"Dropping {col}")
            peaks.drop(col, axis=1, inplace=True)

    # Scale the features without centering
    print("Scaling features...")
    for slide in tqdm(slides, desc="Processing slides"):
        # load scaler
        scaler = joblib.load(f"results/models/scalers/scaler_{slide}.joblib")

        # Fit the scaler on the features
        scaler.fit(peaks.loc[pixels['run'] == slide].values)

        # Transform the features
        peaks.loc[pixels['run'] == slide] = scaler.transform(peaks.loc[pixels['run'] == slide].values)


    # Count the nan values in the peaks dataframe
    n_nan = peaks.isna().sum().sum()
    print(f"Number of NaN values in the peaks dataframe: {n_nan}")

    # Drop the rows with NaN values
    peaks.dropna(axis=0, inplace=True)
    pixels = pixels.loc[peaks.index]

    # reset the index of the peaks dataframe
    peaks.reset_index(drop=True, inplace=True)
    pixels.reset_index(drop=True, inplace=True)

    # Transform the peaks logarithmically
    if FEATURES_TRANSFORM == 'log1p':
        print("Applying logarithmic transformation to peaks...")
        peaks = np.log1p(peaks)

    # Perform dimensionality reduction
    if REDUCTION_N_COMPONENT is not None:
        print(f"Applying {REDUCTION_METHOD.upper()} with n_components={REDUCTION_N_COMPONENT} to features...")
        features_for_dataset = perform_dim_reduction(
            features=peaks,
            n_components=REDUCTION_N_COMPONENT,
            model_base_path=MODEL_BASE_PATH,
            method=REDUCTION_METHOD,
            ica=ICA,
            random_state=42
        )
    else:
        print("No dimensional reduction applied, using standardized features.")
        features_for_dataset = peaks

    # Pass cleaned arrays/DataFrames to the dataset
    print("Creating dataset...")
    dataset = TableDataset(
        features=features_for_dataset,
        target=pixels[TARGET].values,
        target_transform=target_transform
    )

    # Clear memory
    del peaks, features_for_dataset
    gc.collect()

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

    # Inverse transform predictions
    predictions = inverse_transform(predictions)

    # Save predictions
    pixels[f'Predicted_{TARGET}'] = predictions
    output_path = f"results/predictions/mlb_predictions_{MODEL_SUFFIX}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pixels.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Compute Pearson and Spearman correlations between true and predicted targets
    pearson_corr, pearson_p = pearsonr(pixels[TARGET], pixels[f'Predicted_{TARGET}'])
    spearman_corr, spearman_p = spearmanr(pixels[TARGET], pixels[f'Predicted_{TARGET}'])

    print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.2e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")

    # Plot the distribution of the true and predicted targets
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    bins = np.linspace(min(pixels[TARGET].min(), pixels[f'Predicted_{TARGET}'].min()), max(pixels[TARGET].max(), pixels[f'Predicted_{TARGET}'].max()), 100)
    ax.hist(pixels[TARGET], bins=bins, alpha=0.5, label='Data')
    ax.hist(pixels[f'Predicted_{TARGET}'], bins=bins, alpha=0.5, label='Predictions')
    ax.set_title('Prediction vs True Distribution')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    plt.savefig(f"results/figures/mlb_predictions_distribution_{MODEL_SUFFIX}.png")
    plt.close()

    # Plot the predicted CD8 density for each lame compared to the original CD8 density
    fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(25, 40), tight_layout=True)
    ax = axs.flatten()
    for i, lame in tqdm(enumerate(sorted(list(slides) * 2)), desc="Plotting heatmaps"):
        pixels_lame = pixels[pixels['run'] == lame]
        
        if i%2 == 0:
            # Create a pivot table for imshow
            heatmap_data = pixels_lame.pivot(index='y', columns='x', values=TARGET)
            im = ax[i].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=np.quantile(pixels_lame[TARGET], 0.99), origin='upper')
            fig.colorbar(im, ax=ax[i])
        else:
            # Create a pivot table for imshow
            heatmap_data = pixels_lame.pivot(index='y', columns='x', values=f'Predicted_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=np.quantile(pixels_lame[f'Predicted_{TARGET}'], 0.99), origin='upper')
            fig.colorbar(im, ax=ax[i])

        ax[i].set_title(f"{lame} {'Original' if i%2 == 0 else 'Predicted'}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('equal')
        ax[i].invert_yaxis()

    plt.savefig(f"results/figures/mlb_predictions_heatmaps_{MODEL_SUFFIX}.png")