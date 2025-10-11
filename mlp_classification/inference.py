import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import joblib
import gc
import os

from sklearn.metrics import jaccard_score, precision_recall_curve, auc, roc_auc_score, roc_curve, confusion_matrix, classification_report  

from tqdm import tqdm

from dataset import TableDataset
from model import MLPClassifier

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

    PATH = config.get('path_to_results')
    PEAKS_PATH = config.get('peaks_path_inference')
    PIXELS_PATH = config.get('pixels_path_inference')
    TARGET = config.get('target_inference')
    THRESHOLD = config.get('threshold')
    INFERENCE_THRESHOLD = config.get('inference_threshold')  # Default to 0.5 if not provided
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

    # Features Transformations
    FEATURES_TRANSFORM = config.get('features_transform')

    # Dimensionality Reduction Hyperparameters
    REDUCTION_METHOD = config.get('reduction_method')
    REDUCTION_N_COMPONENT = config.get('reduction_n_component')
    ICA = config.get('ica', False)  # Check if ICA is enabled

    # Define model suffix and paths
    MODEL_SUFFIX = f"{THRESHOLD}_{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}_{ARCHITECTURE_FACTOR}_{REDUCTION_N_COMPONENT}_{REDUCTION_METHOD}{'_ica' if ICA else ''}_{LEARNING_RATE}_{WEIGHT_DECAY}"
    MODEL_PATH = f'{PATH}/models/mlp_classification_{MODEL_SUFFIX}.pth'
    PREDICTIONS_PATH = f'{PATH}/predictions/classified_mlp_{MODEL_SUFFIX}.npy'

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load the data
    print("Loading data...")
    peaks_slides = pd.read_feather(PEAKS_PATH)
    pixels_slides = pd.read_feather(PIXELS_PATH)

    # Threshold the target to create a binary classification vector
    pixels_slides[f'Binary_{TARGET}'] = (pixels_slides[TARGET] > THRESHOLD).astype(int)

    # Load dimensionality reduction models
    if REDUCTION_N_COMPONENT is not None:
        print(f"Loading and applying dimensionality reduction model: {REDUCTION_METHOD.upper()} with n_components={REDUCTION_N_COMPONENT}{' + ICA' if ICA else ''}...")

        model_reduction_path = f"{PATH}/models/{REDUCTION_N_COMPONENT}_{REDUCTION_METHOD}.joblib"
        print(f"Loading dimensionality reduction model from {model_reduction_path}")
        model_reduction = joblib.load(model_reduction_path)

        if ICA:
            ica_model_path = f"{PATH}/models/{REDUCTION_N_COMPONENT}_ica.joblib"
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
            scaler = joblib.load(f"{PATH}/models/scalers/scaler_{slide}.joblib")  # load scaler
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
            target=pixels[f'Binary_{TARGET}'].values
        )

        tqdm.write(f"Dataset created with {dataset.n_samples} samples and {dataset.n_features} features.")
        
        # --- Make predictions ---
        tqdm.write("Making predictions...")
        predictions = predict(model, dataset.features, device, batch_size=BATCH_SIZE)

        # Add the predictions to the pixels DataFrame
        tqdm.write("Adding predictions to pixels DataFrame...")
        pixels[f'Classified_{TARGET}'] = predictions
        pixels_all = pd.concat([pixels_all, pixels], ignore_index=True)
        
        # Clear memory
        tqdm.write("Clearing memory...")
        del dataset, peaks, pixels, predictions
        gc.collect()

    # Remove the defects
    print("Removing defects from predictions...")
    pixels_all = pixels_all[~pixels_all['Defects']]
    pixels_all = pixels_all[~pixels_all['Defects']]

    # Subset the data for visualization
    pixels_lesion = pixels_all[pixels_all['Lesion']].copy().reset_index(drop=True)

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(pixels_lesion[f'Binary_{TARGET}'], pixels_lesion[f'Classified_{TARGET}'])

    # Calculate the intersection threshold
    intersection_threshold = thresholds[np.argmin(np.abs(recall - precision))]

    # Calculate AUC for the precision-recall curve
    auc_score = auc(recall, precision)
    print(f"AUC: {auc_score:.2f}")
    print(f"Intersection threshold: {intersection_threshold:.2f}")

    # Threshold the predictions to create a binary classified vector
    if INFERENCE_THRESHOLD:
        inference_threshold = INFERENCE_THRESHOLD
        print(f"Using provided inference threshold: {inference_threshold:.2f}")
    else:
        inference_threshold = intersection_threshold
        print(f"Using intersection threshold as inference threshold: {inference_threshold:.2f}")
    
    pixels_all[f'Binary_classified_{TARGET}'] = (pixels_all[f'Classified_{TARGET}'] > inference_threshold).astype(int)
    pixels_lesion[f'Binary_classified_{TARGET}'] = (pixels_lesion[f'Classified_{TARGET}'] > inference_threshold).astype(int)

    # Save classification predictions to CSV
    output_path = f"{PATH}/predictions/mlp_classification_{TARGET}_{MODEL_SUFFIX}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pixels_all.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    # Classification report
    report_lesion = classification_report(pixels_lesion[f'Binary_{TARGET}'], pixels_lesion[f'Binary_classified_{TARGET}'], target_names=['Low', 'High'])
    report = classification_report(pixels_all[f'Binary_{TARGET}'], pixels_all[f'Binary_classified_{TARGET}'], target_names=['Low', 'High'])
    print("Classification Report (Lesion pixels):\n", report_lesion)
    print("Classification Report (All pixels):\n", report)

    # Compute IOU
    iou = jaccard_score(pixels_all[f'Binary_{TARGET}'], pixels_all[f'Binary_classified_{TARGET}'])
    iou_lesion = jaccard_score(pixels_lesion[f'Binary_{TARGET}'], pixels_lesion[f'Binary_classified_{TARGET}'])
    print(f"IOU (All pixels) at threshold {THRESHOLD}: {iou:.4f}")
    print(f"IOU (Lesion pixels) at threshold {THRESHOLD}: {iou_lesion:.4f}")

    # Compute ROC AUC
    roc_auc = roc_auc_score(pixels_all[f'Binary_{TARGET}'], pixels_all[f'Classified_{TARGET}'])
    roc_auc_lesion = roc_auc_score(pixels_lesion[f'Binary_{TARGET}'], pixels_lesion[f'Classified_{TARGET}'])

    print(f"ROC AUC (All pixels): {roc_auc:.4f}")
    print(f"ROC AUC (Lesion pixels): {roc_auc_lesion:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(pixels_all[f'Binary_{TARGET}'], pixels_all[f'Binary_classified_{TARGET}'])
    cm_lesion = confusion_matrix(pixels_lesion[f'Binary_{TARGET}'], pixels_lesion[f'Binary_classified_{TARGET}'])

    # Compute the roc curve
    fpr, tpr, _ = roc_curve(pixels_all[f'Binary_{TARGET}'], pixels_all[f'Classified_{TARGET}'])
    fpr_lesion, tpr_lesion, _ = roc_curve(pixels_lesion[f'Binary_{TARGET}'], pixels_lesion[f'Classified_{TARGET}'])

    # Plot the confusion matrix
    fig, ax = plt.subplots(3, 2, figsize=(12, 15))

    # Precision-Recall curve (All pixels)
    ax[0, 0].plot(recall, precision, color="purple", label=f"AUC = {auc_score:.2f}")
    ax[0, 0].fill_between(recall, precision, alpha=0.2, color="purple")
    ax[0, 0].set_xlabel("Recall")
    ax[0, 0].set_ylabel("Precision")
    ax[0, 0].set_title("Precision-Recall curve")
    ax[0, 0].legend()

    # Precision & Recall vs Threshold (All pixels)
    ax[0, 1].plot(thresholds, precision[:-1], label="Precision")
    ax[0, 1].plot(thresholds, recall[:-1], label="Recall")
    if INFERENCE_THRESHOLD:
        ax[0, 1].axvline(x=inference_threshold, color='red', linestyle='--', label=f'Inference Threshold {inference_threshold:.2f}')
    ax[0, 1].axvline(x=intersection_threshold, color='purple', linestyle='--', label=f'Intersection Threshold {intersection_threshold:.2f}')
    ax[0, 1].set_xlabel("Threshold")
    ax[0, 1].set_ylabel("Precision & Recall")
    ax[0, 1].set_title("Precision-Recall vs Threshold")
    ax[0, 1].legend()

    # Confusion Matrix (Lesion pixels)
    sns.heatmap(cm_lesion, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1, 0])
    ax[1, 0].set_xlabel('Predicted Label')
    ax[1, 0].set_ylabel('True Label')
    ax[1, 0].set_title('Confusion Matrix (Lesion pixels)')

    # ROC Curve (Lesion pixels)
    ax[1, 1].plot([0, 1], [0, 1], linestyle='--')
    ax[1, 1].plot(fpr_lesion, tpr_lesion, label=f"ROC AUC = {roc_auc_lesion:.2f}")
    ax[1, 1].set_xlabel("False Positive Rate")
    ax[1, 1].set_ylabel("True Positive Rate")
    ax[1, 1].set_title("ROC Curve (Lesion pixels)")
    ax[1, 1].legend()

    # Confusion Matrix (All pixels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[2, 0])
    ax[2, 0].set_xlabel('Predicted Label')
    ax[2, 0].set_ylabel('True Label')
    ax[2, 0].set_title('Confusion Matrix (All pixels)')

    # ROC Curve (All pixels)
    ax[2, 1].plot([0, 1], [0, 1], linestyle='--')
    ax[2, 1].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
    ax[2, 1].set_xlabel("False Positive Rate")
    ax[2, 1].set_ylabel("True Positive Rate")
    ax[2, 1].set_title("ROC Curve (All pixels)")
    ax[2, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{PATH}/figures/mlp_classification_report_{TARGET}_{MODEL_SUFFIX}.png")

    # Plot the classified target density for each slide compared to the original target density
    fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(25, 40), tight_layout=True)
    ax = axs.flatten()
    for i, slide in tqdm(enumerate(sorted(list(slides) * 2)), desc="Plotting heatmaps"):
        pixels_slide = pixels_all[pixels_all['batch'] == slide]
        if i % 2 == 0:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=f'Binary_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', origin='upper')
        else:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=f'Binary_classified_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', origin='upper')

        ax[i].set_title(f"{slide} {'Original' if i%2 == 0 else 'Classified'}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('equal')
        ax[i].invert_yaxis()

    plt.savefig(f"{PATH}/figures/mlp_classification_heatmaps_{TARGET}_{MODEL_SUFFIX}.png")

    fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(25, 40), tight_layout=True)
    ax = axs.flatten()
    for i, slide in tqdm(enumerate(sorted(list(slides) * 2)), desc="Plotting heatmaps"):
        pixels_slide = pixels_lesion[pixels_lesion['batch'] == slide]
        if i % 2 == 0:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=f'Binary_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', origin='upper')
        else:
            # Create a pivot table for imshow
            heatmap_data = pixels_slide.pivot(index='y', columns='x', values=f'Binary_classified_{TARGET}')
            im = ax[i].imshow(heatmap_data, cmap='viridis', origin='upper')

        ax[i].set_title(f"{slide} {'Original' if i%2 == 0 else 'Classified'}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('equal')
        ax[i].invert_yaxis()

    plt.savefig(f"{PATH}/figures/mlp_classification_heatmaps_{TARGET}_{MODEL_SUFFIX}_lesion.png")