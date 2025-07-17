import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import yaml
import numpy as np
import pandas as pd

# Import custom modules
from train import train_xgb_model
from utils import get_target_transform, get_inverse_transform

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# Load configuration from YAML file
with open("xgb_regression/config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)  # Load model configuration from YAML file

# Set the device
device = config.get('device')

# Extract hyperparameters from the config
PATH = config.get('path_to_data')
PEAKS_PATH = config.get('peaks_path')
PIXELS_PATH = config.get('pixels_path')
TARGET = config.get('target')
TARGET_THRESHOLD = config.get('target_threshold')
EXCLUDED_SLIDES = config.get('excluded_slides')

# Target Transformation Hyperparameters
TARGET_TRANSFORM = config.get('target_transform')

# Save paths
MODEL_SUFFIX = f"{TARGET}_{TARGET_TRANSFORM}_{config.get('objective')}_{config.get('max_depth')}_{config.get('learning_rate')}_{config.get('alpha')}_{config.get('colsample_bytree')}"
MODEL_SAVE_PATH = f'results/models/xgb_regressor_{MODEL_SUFFIX}.joblib'
PLOT_LOSS_PATH = f'results/figures/xgb_regression_loss_{MODEL_SUFFIX}.png'

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

# Scale the features without centering
print("Scaling features...")
scaler = StandardScaler(with_mean=False)  # initialize scaler without centering
scaler.fit(peaks.values)  # fit the scaler on the features
peaks = pd.DataFrame(scaler.transform(peaks.values), columns=peaks.columns)  # scale the features and convert it back to DataFrame

# Save the scaler model
joblib.dump(scaler, f"xgb_{MODEL_SUFFIX}_scaler.joblib")

# Transform the target using the transformation function
target_transform = get_target_transform(TARGET_TRANSFORM)
inverse_transform = get_inverse_transform(TARGET_TRANSFORM)

# Apply the target transformation
pixels[f'{TARGET}_transformed'] = target_transform(pixels[TARGET])

# Split dataset into training and validation sets
print("Splitting dataset into training and validation sets...")
X_train, X_test, y_train, y_test = train_test_split(peaks,
                                                    pixels[f'{TARGET}_transformed'],
                                                    test_size=config.get('validation_split'),
                                                    random_state=42)
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_test)}")

# Prepare XGBoost parameters
params = {
    'objective': config.get('objective'),
    'max_depth': config.get('max_depth'),
    'learning_rate': config.get('learning_rate'),
    'alpha': config.get('alpha'),
    'colsample_bytree': config.get('colsample_bytree'),
    'scale_pos_weight': float(np.sum(pixels[TARGET] < TARGET_THRESHOLD) / np.sum(pixels[TARGET] > TARGET_THRESHOLD)),
    'device': config.get('device')
}

evals_result = {}

# Train XGBoost model
model_reg, r2_train, r2_test, mse_train, mse_test = train_xgb_model(
    X_train, X_test, y_train, y_test,
    params,
    evals_result,
    MODEL_SAVE_PATH,
    config.get('num_boost_rounds'),
    config.get('early_stopping_rounds'),
    PLOT_LOSS_PATH,
    inverse_transform
)