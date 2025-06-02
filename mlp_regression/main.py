import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
import os  # For checking if model file exists
import yaml

# Import custom modules
from dataset import MALDI_multisamples
from model import MLPRegression
from train import train_model
from inference import predict

# --- Configuration ---
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
MAX_LR = config.get('max_lr')
WEIGHT_DECAY = config.get('weight_decay')

# Early Stopping Hyperparameters
PATIENCE = config.get('patience')
MIN_DELTA = config.get('min_delta')

# MLP Hyperparameters
HIDDEN_DIM = config.get('hidden_dim')
NUM_HIDDEN_LAYERS = config.get('num_hidden_layers')
MODEL_SAVE_PATH = f'results/models/MLP_regression_{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}.pth'
PLOT_LOSS_PATH = f'results/figures/MLP_regression_loss_{HIDDEN_DIM}_{NUM_HIDDEN_LAYERS}.png'

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Data
print("Loading data...")
dataset = MALDI_multisamples(features=PEAKS_PATH, targets=PIXELS_PATH, target=TARGET, excluded_slides=EXCLUDED_SLIDES)

# Split dataset into training and validation sets
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
model = MLPRegression(input_dim=input_dim, output_dim=output_dim, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS)

# Loss Function (Huber Loss for regression)
criterion = nn.HuberLoss(delta=HUBER_DELTA) 

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# OneCycleLR Scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS
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

# Inference Example
print("\n--- Inference Example ---")

# Create a new model instance for loading (or reuse the trained 'model')
inference_model = MLPRegression(input_dim=input_dim, output_dim=output_dim, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS)

# Load the saved state dictionary
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading model weights from {MODEL_SAVE_PATH}...")
    inference_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    inference_model.to(device)  # Move model to device *after* loading weights

    # Prepare some sample data for inference (e.g., the first 3 samples from the dataset)
    sample_features, _ = dataset[0:3]  # Get features for first 3 samples
    print(f"\nSample features for prediction:\n{sample_features}")

    # Make predictions
    predictions = predict(inference_model, sample_features, device)
    print(f"\nPredictions:\n{predictions}")

    # Optional: Print corresponding actual targets
    _, actual_targets = dataset[0:3]
    print(f"\nActual Targets:\n{actual_targets.numpy()}")  # Convert targets tensor to numpy for printing
else:
    print(f"Model file not found at {MODEL_SAVE_PATH}. Skipping inference.")

print("\nScript finished.")