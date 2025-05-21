import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os  # For checking if model file exists

# Import custom modules
from dataset import MALDI_multisamples
from model import LinearRegression
from train import train_model
from inference import predict

# --- Configuration ---
PATH = "data/MALDI_IHC/correlations/"
PEAKS_PATH = f"{PATH}peaks_standardized_lasso.pkl"
PIXELS_PATH = f"{PATH}pixels_filtered_lasso.pkl"
TARGET = 'Density_CD8'
MODEL_SAVE_PATH = 'models/linear_regression.pth'
LEARNING_RATE = 0.01
NUM_EPOCHS = 100
BATCH_SIZE = 10**5
VALIDATION_SPLIT = 0.1  # Fraction of data to use for validation

# Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 1. Load Data ---
print("Loading data...")
dataset = MALDI_multisamples(peaks=PEAKS_PATH, pixels=PIXELS_PATH, target=TARGET)

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

# --- 2. Initialize Model, Loss, Optimizer ---
print("Initializing model...")
model = LinearRegression(input_dim=input_dim, output_dim=output_dim)

# Loss Function (Mean Squared Error for regression)
criterion = nn.MSELoss()

# Optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
# You could also use Adam: optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 3. Training ---
print("Starting training...")
history = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, plot_loss=True)

# --- 4. Save the Trained Model ---
print(f"Saving model to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Model saved.")

# --- 5. Inference Example ---
print("\n--- Inference Example ---")

# Create a new model instance for loading (or reuse the trained 'model')
inference_model = LinearRegression(input_dim=input_dim, output_dim=output_dim)

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