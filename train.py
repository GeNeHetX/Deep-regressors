# train.py
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                num_epochs: int=10,
                device: torch.device=torch.device('cpu'),
                plot_loss: bool=False) -> dict:
    """
    Trains the PyTorch model and validates after each epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader providing batches of training data.
        val_loader (DataLoader): DataLoader providing batches of validation data.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): The device to train on ('cpu' or 'cuda').
        plot_loss (bool): Whether to plot the training and validation loss.

    Returns:
        dict: A dictionary containing training and validation losses for each epoch.
    """
    model.to(device)  # Move model to the specified device
    print(f"Starting training on {device} for {num_epochs} epochs...")

    # Initialize history dictionary
    history = {'train_loss': [], 'val_loss': []}

    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        # Training phase
        model.train()  # Set the model to training mode
        epoch_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        # Print losses for the epoch
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.2e}, Val Loss: {avg_val_loss:.2e}")

    print("Training finished.")

    # Optionally plot the losses
    if plot_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig('training_validation_loss.png')
        plt.close()

    return history  # Return the history dictionary

# Example usage (optional, part of main.py usually)
if __name__ == '__main__':
    # This part is typically run from main.py
    print("This script contains the training function.")
    print("To run training, execute main.py.")