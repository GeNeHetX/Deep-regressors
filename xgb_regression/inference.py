import xgboost as xgb
import joblib
import yaml
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# Import custom modules
from utils import get_target_transform, get_inverse_transform

# Load configuration from YAML file
with open("xgb_regression/config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)

# Extract hyperparameters from the config
PATH = config.get('path_to_data')
PEAKS_PATH = config.get('peaks_path')
PIXELS_PATH = config.get('pixels_path')
TARGET = config.get('target')
TARGET_TRANSFORM = config.get('target_transform')
EXCLUDED_SLIDES = config.get('excluded_slides')

# Model and scaler paths
MODEL_SUFFIX = f"{TARGET}_{TARGET_TRANSFORM}_{config.get('objective')}_{config.get('max_depth')}_{config.get('learning_rate')}_{config.get('alpha')}_{config.get('colsample_bytree')}"
MODEL_LOAD_PATH = f'results/models/xgb_regressor_{MODEL_SUFFIX}.joblib'
SCALER_PATH = f"xgb_{MODEL_SUFFIX}_scaler.joblib"

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

# Drop the peaks that are in the trypsin peptide masses with tolerance 0.2
with open("trypsin_peaks.yaml", "r") as f:
    trypsin_peaks = yaml.safe_load(f)

for col in peaks.columns:
    if np.min(np.abs(float(col) - np.array(trypsin_peaks))) < 0.2:
        print(f"Dropping {col}")
        peaks.drop(col, axis=1, inplace=True)

# Load scaler and scale features
print("Loading scaler and scaling features...")
scaler = joblib.load(SCALER_PATH)
peaks_scaled = pd.DataFrame(scaler.transform(peaks.values), columns=peaks.columns)

# Load XGBoost model
print("Loading XGBoost model...")
model_reg = joblib.load(MODEL_LOAD_PATH)

# Prepare DMatrix for inference
dall = xgb.DMatrix(peaks_scaled)

# Run inference
print("Running inference on all data...")
y_pred = model_reg.predict(dall)
y_pred = np.clip(y_pred, 0, 1)

# Inverse transform predictions
inverse_transform = get_inverse_transform(TARGET_TRANSFORM)
y_pred_inverse = inverse_transform(y_pred)

# Save predictions
pixels[f'Predicted_{TARGET}'] = y_pred_inverse
output_path = f"results/predictions/xgb_predictions_{MODEL_SUFFIX}.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pixels.to_csv(output_path, index=False)

# Compute Pearson and Spearman correlations between true and predicted targets
pearson_corr, pearson_p = pearsonr(pixels[TARGET], pixels[f'Predicted_{TARGET}'])
spearman_corr, spearman_p = spearmanr(pixels[TARGET], pixels[f'Predicted_{TARGET}'])

print(f"Predictions saved to {output_path}")
print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.2e})")
print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")

# Plot the predicted CD8 density for each lame compared to the original CD8 density
fig, axs = plt.subplots(nrows=11, ncols=6, figsize=(25, 40), tight_layout=True)
ax = axs.flatten()
for i, lame in enumerate(sorted(list(pixels['run'].unique()) * 2)):
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

plt.savefig(f"results/figures/xgb_predictions_heatmaps_{MODEL_SUFFIX}.png")