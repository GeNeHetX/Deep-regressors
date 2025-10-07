import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import joblib
import gc
import os

from sklearn.metrics import jaccard_score
from tqdm import tqdm

from dataset import TableDataset
from model import MLPClassifier
from utils import get_target_transform, get_inverse_transform

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
    with open("mlp_classification/config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)  # Load model configuration from YAML file

    PATH = config.get('path_to_data_inference')
    PEAKS_PATH = config.get('peaks_path_inference')
    PIXELS_PATH = config.get('pixels_path_inference')
    TARGET = config.get('target_inference')
    THRESHOLD = config.get('threshold')
    EXCLUDED_SLIDES = config.get('excluded_slides')  # Default to empty list if not provided
    SCALE = config.get('scale')
    BATCH_SIZE = config.get('batch_size_inference')

    # Optimization Hyperparameters
    LEARNING_RATE = config.get('learning_rate')
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
    MODEL_SUFFIX = f"{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}_{ARCHITECTURE_FACTOR}_{REDUCTION_N_COMPONENT}_{REDUCTION_METHOD}{'_ica' if ICA else ''}_{LEARNING_RATE}_{WEIGHT_DECAY}"
    MODEL_PATH = f'results/models/MLP_classification_{MODEL_SUFFIX}.pth'
    PREDICTIONS_PATH = f'results/predictions/classified_mlp_{MODEL_SUFFIX}.npy'

    # Define target transformation functions
    target_transform = get_target_transform(TARGET_TRANSFORM)
    inverse_transform = get_inverse_transform(TARGET_TRANSFORM)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load the data
    print("Loading data...")
    peaks_slides = pd.read_feather(PEAKS_PATH)
    pixels_slides = pd.read_feather(PIXELS_PATH)

    # Load dimensionality reduction models
    if REDUCTION_N_COMPONENT is not None:
        print(f"Loading and applying dimensionality reduction model: {REDUCTION_METHOD.upper()} with n_components={REDUCTION_N_COMPONENT}{' + ICA' if ICA else ''}...")

        model_reduction_path = f"results/models/{REDUCTION_N_COMPONENT}_{REDUCTION_METHOD}.joblib"
        print(f"Loading dimensionality reduction model from {model_reduction_path}")
        model_reduction = joblib.load(model_reduction_path)

        if ICA:
            ica_model_path = f"results/models/{REDUCTION_N_COMPONENT}_ica.joblib"
            print(f"Loading ICA model from {ica_model_path}")
            ica_model = joblib.load(ica_model_path)

    # --- Load the model ---
    input_dim = peaks_slides.shape[1]
    output_dim = 1 

    print("Loading the model...")
    model = MLPClassifier(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=HIDDEN_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        architecture_factor=ARCHITECTURE_FACTOR,
        dropout=DROPOUT
    )
    print(model)
    
    model.load_state_dict(torch.load(MODEL_PATH))

    # Move model to the appropriate device
    model.to(device)

    # List all the slides in the data directory, excluding the ones in excluded_slides
    slides = np.sort([slide for slide in os.listdir("data_external") if slide not in EXCLUDED_SLIDES])
    n_slides = len(slides)
    print(f"Found {n_slides} slides for inference.")

    # Load trypsin peaks
    with open("trypsin_peaks.yaml", "r") as f:
        trypsin_peaks = yaml.safe_load(f)

    pixels_all = pd.DataFrame()  # Initialize an empty DataFrame to collect all predictions

    for slide in tqdm(slides, desc="Processing slides"):
        tqdm.write(f"Processing slide: {slide}")

        # Load Data
        tqdm.write("Loading data...")
        peaks = peaks_slides[pixels_slides['batch'] == slide].copy().reset_index(drop=True)
        pixels = pixels_slides[pixels_slides['batch'] == slide].copy().reset_index(drop=True)
        # peaks = pd.read_pickle(f"data_external/{slide}/results/peaks_aligned.pkl")
        # pixels = pd.read_feather(f"data_external/{slide}/results/mse_pixels.feather")
        # pixels.rename(columns={'run': 'batch'}, inplace=True)

        # # Drop the peaks that are in the trypsin peptide masses with tolerance 0.2
        # for col in peaks.columns:
        #     if np.min(np.abs(float(col) - np.array(trypsin_peaks))) < 0.2:
        #         tqdm.write(f"Dropping trypsin peak: {col}")
        #         peaks.drop(col, axis=1, inplace=True)

        # # Drop microdissection columns from pixels DataFrame
        # pixels = pixels.drop(columns=pixels.filter(regex='Density_microdissection_').columns)

        # Scale the features without centering
        if SCALE:
            tqdm.write("Scaling features...")
            scaler = joblib.load(f"results/models/scalers/scaler_{slide}.joblib")  # load scaler
            peaks.loc[pixels['batch'] == slide] = scaler.transform(peaks.loc[pixels['batch'] == slide].values)  # Transform the features

        # Count the nan values in the peaks dataframe
        n_nan = peaks.isna().sum().sum()
        tqdm.write(f"Number of NaN values in the peaks dataframe: {n_nan}")

        # Drop the rows with NaN values
        peaks.dropna(axis=0, inplace=True)
        pixels = pixels.loc[peaks.index]

        # reset the index of the peaks dataframe
        peaks.reset_index(drop=True, inplace=True)
        pixels.reset_index(drop=True, inplace=True)

        # Transform the peaks logarithmically
        if FEATURES_TRANSFORM == 'log1p':
            tqdm.write("Applying logarithmic transformation to peaks...")
            peaks = np.log1p(peaks)

        # Apply dimensionality reduction if specified
        if REDUCTION_N_COMPONENT is not None:
            tqdm.write(f"Applying dimensionality reduction: {REDUCTION_METHOD.upper()} with n_components={REDUCTION_N_COMPONENT}{' + ICA' if ICA else ''}...")

            # Apply the dimensionality reduction model
            peaks = model_reduction.transform(peaks)

            # If ICA is enabled, apply ICA transformation
            if ICA:
                tqdm.write("Applying ICA transformation...")
                peaks = ica_model.transform(peaks)

        # Pass cleaned arrays/DataFrames to the dataset
        tqdm.write("Creating dataset...")
        dataset = TableDataset(
            features=peaks.values,
            target=np.where(pixels[TARGET].values > THRESHOLD, 1, 0),  # Binarize target based on threshold
            target_transform=target_transform
        )

        tqdm.write(f"Dataset created with {dataset.n_samples} samples and {dataset.n_features} features.")
        
        # --- Make predictions ---
        tqdm.write("Making predictions...")
        predictions = predict(model, dataset.features, device, batch_size=BATCH_SIZE)

        # Inverse transform predictions
        tqdm.write(f"Applying inverse transformation to predictions using {TARGET_TRANSFORM}...")
        predictions = inverse_transform(predictions)

        # Add the predictions to the pixels DataFrame
        tqdm.write("Adding predictions to pixels DataFrame...")
        pixels[f'classified_{TARGET}'] = predictions
        pixels_all = pd.concat([pixels_all, pixels], ignore_index=True)
        
        # Clear memory
        tqdm.write("Clearing memory...")
        del dataset, peaks, pixels, predictions
        gc.collect()

    # Remove the defects
    print("Removing defects from predictions...")
    pixels_all = pixels_all[~pixels_all['Defects']]
    pixels_all = pixels_all[~pixels_all['Defects']]

    # Save classification predictions to CSV
    output_path = f"results/predictions/mlb_classification_{TARGET}_{MODEL_SUFFIX}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pixels_all.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Subset the data for visualization
    pixels_lesion = pixels_all[pixels_all['Lesion']]

    # Compute IOU
    y_true = (pixels_all[TARGET] >= THRESHOLD).astype(int)
    y_pred = (pixels_all[f'classified_{TARGET}'] >= THRESHOLD).astype(int)
    iou = jaccard_score(y_true, y_pred)
    print(f"IOU (All pixels) at threshold {THRESHOLD}: {iou:.4f}")

    y_true_lesion = (pixels_lesion[TARGET] >= THRESHOLD).astype(int)
    y_pred_lesion = (pixels_lesion[f'classified_{TARGET}'] >= THRESHOLD).astype(int)
    iou_lesion = jaccard_score(y_true_lesion, y_pred_lesion)
    print(f"IOU (Lesion pixels) at threshold {THRESHOLD}: {iou_lesion:.4f}")

    # Plot the classified target density for each slide compared to the original target density
    fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(25, 40), tight_layout=True)
    ax = axs.flatten()
    for i, slide in tqdm(enumerate(sorted(list(slides) * 2)), desc="Plotting heatmaps"):
        pixels_slide = pixels_all[pixels_all['batch'] == slide]
        if i % 2 == 0:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=TARGET)
            im = ax[i].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=np.quantile(pixels_slide[TARGET], 0.99), origin='upper')
            fig.colorbar(im, ax=ax[i])
        else:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=f'classified_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', origin='upper')
            fig.colorbar(im, ax=ax[i])

        ax[i].set_title(f"{slide} {'Original' if i%2 == 0 else 'Classified'}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('equal')
        ax[i].invert_yaxis()

    plt.savefig(f"results/figures/mlb_classification_heatmaps_{TARGET}_{MODEL_SUFFIX}.png")

    fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(25, 40), tight_layout=True)
    ax = axs.flatten()
    for i, slide in tqdm(enumerate(sorted(list(slides) * 2)), desc="Plotting heatmaps"):
        pixels_slide = pixels_lesion[pixels_lesion['batch'] == slide]
        if i % 2 == 0:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=TARGET)
            im = ax[i].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=np.quantile(pixels_slide[TARGET], 0.99), origin='upper')
            fig.colorbar(im, ax=ax[i])
        else:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=f'classified_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', origin='upper')
            fig.colorbar(im, ax=ax[i])

        ax[i].set_title(f"{slide} {'Original' if i%2 == 0 else 'Classified'}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('equal')
        ax[i].invert_yaxis()

    plt.savefig(f"results/figures/mlb_classification_heatmaps_{TARGET}_{MODEL_SUFFIX}_lesion.png")