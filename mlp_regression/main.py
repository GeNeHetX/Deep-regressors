import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import yaml
import numpy as np
import pandas as pd
import os
import gc

# Import custom modules
from dataset import TableDataset
from model import MLPRegression
from train import train_model
from utils import get_target_transform, get_inverse_transform, perform_dim_reduction

# Load configuration from YAML file
with open("mlp_regression/config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)  # Load model configuration from YAML file

# Extract hyperparameters from the config
PATH = config.get('path_to_data')
PEAKS_PATH = config.get('peaks_path')
PIXELS_PATH = config.get('pixels_path')
TARGET = config.get('target')
EXCLUDED_SLIDES = config.get('excluded_slides')

# Training Hyperparameters
NUM_EPOCHS = config.get('num_epochs')
BATCH_SIZE = config.get('batch_size')
VALIDATION_SPLIT = config.get('validation_split')

# Optimization Hyperparameters
HUBER_DELTA = config.get('huber_delta')
LEARNING_RATE = config.get('learning_rate')
WEIGHT_DECAY = config.get('weight_decay')
SCHEDULER_FACTOR = config.get('scheduler_factor')
SCHEDULER_PATIENCE = config.get('scheduler_patience')

# Early Stopping Hyperparameters
PATIENCE = config.get('patience')
MIN_DELTA = config.get('min_delta')

# MLP Hyperparameters
HIDDEN_DIM = config.get('hidden_dim')
NUM_HIDDEN_LAYERS = config.get('num_hidden_layers')
ARCHITECTURE_FACTOR = config.get('architecture_factor')
DROPOUT = config.get('dropout')

# Target and Features Transformations
TARGET_TRANSFORM = config.get('target_transform')
FEATURES_TRANSFORM = config.get('features_transform')

# Dimensionality Reduction Hyperparameters
REDUCTION_METHOD = config.get('reduction_method')
REDUCTION_N_COMPONENT = config.get('reduction_n_component')
ICA = config.get('ica', False)  # Check if ICA is enabled

# Define model suffix and paths
MODEL_SUFFIX = f"{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}_{ARCHITECTURE_FACTOR}_{REDUCTION_N_COMPONENT}_{REDUCTION_METHOD}{'_ica' if ICA else ''}_{HUBER_DELTA}_{LEARNING_RATE}_{WEIGHT_DECAY}"
MODEL_SAVE_PATH = f'results/models/MLP_regression_{MODEL_SUFFIX}.pth'
PLOT_LOSS_PATH = f'results/figures/MLP_regression_loss_{MODEL_SUFFIX}.png'
MODEL_BASE_PATH = f"results/models/{REDUCTION_N_COMPONENT}"

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

# Scale the features without centering
print("Scaling features...")
for slide in tqdm(slides, desc="Processing slides"):
    # Check if the slide exists in pixels
    try:
        # Load the scaler for the current slide
        scaler = joblib.load(f"results/models/scalers/scaler_{slide}.joblib")
    except FileNotFoundError:
        # Initialize scaler without centering
        scaler = StandardScaler(with_mean=False, with_std=True)

        # Fit the scaler on the features
        scaler.fit(peaks.loc[pixels['run'] == slide].values)

        # Save the scaler
        joblib.dump(scaler, f"results/models/scalers/scaler_{slide}.joblib")

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

print(f"Dataset created with {dataset.n_samples} samples and {dataset.n_features} features.")

# Clear memory
del peaks, pixels, features_for_dataset
gc.collect()

# Split dataset into training and validation sets
print("Splitting dataset into training and validation sets...")
dataset_size = len(dataset)
val_size = int(VALIDATION_SPLIT * dataset_size)
train_size = dataset_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Get input dimension from dataset
input_dim = dataset.n_features
output_dim = 1

# Initialize Model, Loss and Optimizer
print("Initializing model...")
model = MLPRegression(
    input_dim=input_dim,
    output_dim=output_dim,
    hidden_dim=HIDDEN_DIM,
    num_hidden_layers=NUM_HIDDEN_LAYERS,
    architecture_factor=ARCHITECTURE_FACTOR,
    dropout=DROPOUT
)
print(model)

# Loss Function (Huber Loss for regression)
criterion = nn.HuberLoss(delta=HUBER_DELTA) 

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# OneCycleLR Scheduler
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',      # Reduce LR when the metric has stopped decreasing
    factor=SCHEDULER_FACTOR,      # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=SCHEDULER_PATIENCE      # Number of epochs with no improvement after which learning rate will be reduced
)

# Training
print("Starting training...")
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=NUM_EPOCHS,
    device=device,
    patience=PATIENCE,
    min_delta=MIN_DELTA,
    scheduler=scheduler,
    plot_loss_path=PLOT_LOSS_PATH,
    model_save_path=MODEL_SAVE_PATH
)
print("Training completed.")

# Run inference.py script to generate predictions
print("Running inference...")
os.system(f"python mlp_regression/inference.py")