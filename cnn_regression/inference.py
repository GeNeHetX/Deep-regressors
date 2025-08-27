import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import joblib
import gc
import os

from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from dataset import MSI_Image_Dataset
from model import UNet
from utils import get_target_transform, get_inverse_transform

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

def predict(model: nn.Module,
            data_tensor: torch.Tensor,
            device: torch.device,
            batch_size: int = 1) -> np.ndarray:
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
    with open("cnn_regression/config.yaml", 'r') as config_file:
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

    # Target and Features Transformations
    TARGET_TRANSFORM = config.get('target_transform')
    FEATURES_TRANSFORM = config.get('features_transform')

    # Dimensionality Reduction Hyperparameters
    REDUCTION_METHOD = config.get('reduction_method')
    REDUCTION_N_COMPONENT = config.get('reduction_n_component')
    ICA = config.get('ica', False)  # Check if ICA is enabled

    # Define model suffix and paths
    MODEL_SUFFIX = f"{REDUCTION_N_COMPONENT}_{REDUCTION_METHOD}{'_ica' if ICA else ''}_{HUBER_DELTA}_{LEARNING_RATE}_{MAX_LR}_{WEIGHT_DECAY}"
    MODEL_PATH = f'results/models/UNet_regression_{MODEL_SUFFIX}.pth'
    PREDICTIONS_PATH = f'results/predictions/predictions_unet_{MODEL_SUFFIX}.npy'

    # Define target transformation functions
    target_transform = get_target_transform(TARGET_TRANSFORM)
    inverse_transform = get_inverse_transform(TARGET_TRANSFORM)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    in_channels = REDUCTION_N_COMPONENT

    print("Loading the model...")
    model = UNet(in_channels=in_channels)
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
        peaks = pd.read_pickle(f"data_external/{slide}/results/peaks_aligned.pkl")
        pixels = pd.read_feather(f"data_external/{slide}/results/mse_pixels.feather")

        # Drop the peaks that are in the trypsin peptide masses with tolerance 0.2
        for col in peaks.columns:
            if np.min(np.abs(float(col) - np.array(trypsin_peaks))) < 0.2:
                tqdm.write(f"Dropping trypsin peak: {col}")
                peaks.drop(col, axis=1, inplace=True)

        # Drop microdissection columns from pixels DataFrame
        pixels = pixels.drop(columns=pixels.filter(regex='Density_microdissection_').columns)

        # Scale the features without centering
        tqdm.write("Scaling features...")
        scaler = joblib.load(f"results/models/scalers/scaler_{slide}.joblib")  # load scaler
        scaler.fit(peaks.loc[pixels['run'] == slide].values)  # Fit the scaler on the features
        peaks.loc[pixels['run'] == slide] = scaler.transform(peaks.loc[pixels['run'] == slide].values)  # Transform the features

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
            features_for_dataset = pd.DataFrame(model_reduction.transform(peaks))

            # If ICA is enabled, apply ICA transformation
            if ICA:
                tqdm.write("Applying ICA transformation...")
                features_for_dataset = pd.DataFrame(ica_model.transform(features_for_dataset.values))

        # Pass cleaned arrays/DataFrames to the dataset
        tqdm.write("Creating dataset...")
        dataset = MSI_Image_Dataset(
            features=features_for_dataset,
            coordinates=pixels[['x', 'y']],
            samples_indices=pixels['run'].values,
            target=pixels[TARGET].values
        )

        tqdm.write(f"Dataset created with {dataset.n_observations} samples and {dataset.n_features} features.")

        # --- Make predictions ---
        tqdm.write("Making predictions...")
        sample_img, _, _ = dataset[0]
        predictions = predict(model, sample_img, device, batch_size=BATCH_SIZE)

        # Inverse transform predictions
        tqdm.write(f"Applying inverse transformation to predictions using {TARGET_TRANSFORM}...")
        predictions = inverse_transform(predictions)

        # Add the predictions to the pixels DataFrame
        tqdm.write("Adding predictions to pixels DataFrame...")
        pixels[f'Predicted_{TARGET}'] = predictions
        pixels_all = pd.concat([pixels_all, pixels], ignore_index=True)
        
        # Clear memory
        tqdm.write("Clearing memory...")
        del dataset, peaks, features_for_dataset, pixels, predictions
        gc.collect()

    # Remove the defects
    print("Removing defects from predictions...")
    pixels_all = pixels_all[pixels_all['Density_Defects'] < 0.1]
    pixels_all = pixels_all[pixels_all['Density_Defects'] < 0.1]

    # Save predictions
    output_path = f"results/predictions/unet_predictions_{TARGET}_{MODEL_SUFFIX}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pixels_all.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Subset the data for visualization
    pixels_lesion = pixels_all[pixels_all['Density_Lesion'] > 0.5]

    # Compute Pearson and Spearman correlations between true and predicted targets
    pearson_corr, pearson_p = pearsonr(pixels_lesion[TARGET], pixels_lesion[f'Predicted_{TARGET}'])
    spearman_corr, spearman_p = spearmanr(pixels_lesion[TARGET], pixels_lesion[f'Predicted_{TARGET}'])

    print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.2e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")

    # Plot the distribution of the true and predicted targets
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    bins = np.linspace(min(pixels_lesion[TARGET].min(), pixels_lesion[f'Predicted_{TARGET}'].min()), max(pixels_lesion[TARGET].max(), pixels_lesion[f'Predicted_{TARGET}'].max()), 100)
    ax.hist(pixels_lesion[TARGET], bins=bins, alpha=0.5, label='Data')
    ax.hist(pixels_lesion[f'Predicted_{TARGET}'], bins=bins, alpha=0.5, label='Predictions')
    ax.set_title('Prediction vs True Distribution')
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    plt.savefig(f"results/figures/unet_predictions_distribution_{TARGET}_{MODEL_SUFFIX}.png")
    plt.close()

    # Plot the predicted target density for each slide compared to the original target density
    fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(25, 40), tight_layout=True)
    ax = axs.flatten()
    for i, slide in tqdm(enumerate(sorted(list(slides) * 2)), desc="Plotting heatmaps"):
        pixels_slide = pixels_all[pixels_all['run'] == slide]
        if i % 2 == 0:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=TARGET)
            im = ax[i].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=np.quantile(pixels_slide[TARGET], 0.99), origin='upper')
            fig.colorbar(im, ax=ax[i])
        else:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=f'Predicted_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=np.quantile(pixels_slide[f'Predicted_{TARGET}'], 0.99), origin='upper')
            fig.colorbar(im, ax=ax[i])

        ax[i].set_title(f"{slide} {'Original' if i%2 == 0 else 'Predicted'}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('equal')
        ax[i].invert_yaxis()

    plt.savefig(f"results/figures/unet_predictions_heatmaps_{TARGET}_{MODEL_SUFFIX}.png")

    fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(25, 40), tight_layout=True)
    ax = axs.flatten()
    for i, slide in tqdm(enumerate(sorted(list(slides) * 2)), desc="Plotting heatmaps"):
        pixels_slide = pixels_lesion[pixels_lesion['run'] == slide]
        if i % 2 == 0:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=TARGET)
            im = ax[i].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=np.quantile(pixels_slide[TARGET], 0.99), origin='upper')
            fig.colorbar(im, ax=ax[i])
        else:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=f'Predicted_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', vmin=0, vmax=np.quantile(pixels_slide[f'Predicted_{TARGET}'], 0.99), origin='upper')
            fig.colorbar(im, ax=ax[i])

        ax[i].set_title(f"{slide} {'Original' if i%2 == 0 else 'Predicted'}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('equal')
        ax[i].invert_yaxis()

    plt.savefig(f"results/figures/unet_predictions_heatmaps_{TARGET}_{MODEL_SUFFIX}_lesion.png")