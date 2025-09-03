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
                dataset: torch.utils.data.Dataset,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                validation_split: float=0.2,
                batch_size: int=32,
                num_epochs: int=10,
                device: torch.device=torch.device('cpu'),
                plot_loss_path: str=None,
                patience: int=10,
                min_delta: float=1e-6,
                scheduler=None,
                model_save_path: str=None) -> dict:
    """
    Trains the PyTorch model and validates after each epoch, with early stopping.

    Args:
        model (nn.Module): The model to train.
        dataset: The full dataset to split and train on.
        criterion: The loss function.
        optimizer: The optimization algorithm.
        validation_split (float): Fraction of data to use for validation.
        batch_size (int): Batch size for DataLoaders.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): The device to train on ('cpu' or 'cuda').
        plot_loss_path (str): Path to save the training and validation loss plot.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
        scheduler: Learning rate scheduler (optional).
        model_save_path (str): Path to save the best model.

    Returns:
        dict: A dictionary containing training and validation losses for each epoch.
    """
    # Split dataset into training and validation sets
    print("Splitting dataset into training and validation sets...")
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create DataLoaders for the training and validation sets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    print(f"DataLoaders created with lengths: {len(train_loader)}, {len(val_loader)}.")

    model.to(device)  # Move model to the specified device
    print(f"Starting training on {device} for {num_epochs} epochs...")

    # Initialize history dictionary
    history = {'train_loss': [], 'val_loss': []}

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_train_loss += loss.item() * inputs.size(0)

        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item() * inputs.size(0)

        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        history['val_loss'].append(avg_val_loss)

        # Print losses for the epoch
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.2e}, Val Loss: {avg_val_loss:.2e}")

        # Early stopping logic
        if epoch > 0:
            # Check if validation loss is not decreasing enough
            if history['val_loss'][-2] - avg_val_loss < min_delta:
                patience_counter += 1

            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                patience_counter = 0
                if model_save_path:
                    torch.save(model.state_dict(), model_save_path)

            if patience_counter >= patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1} with val loss {best_val_loss:.2e}.")
                break

    print("Training finished.")

    # Optionally plot the losses
    if plot_loss_path:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.yscale('log')
        plt.savefig(plot_loss_path)
        plt.close()

    return history  # Return the history dictionary

# Example usage (optional, part of main.py usually)
if __name__ == '__main__':
    # This part is typically run from main.py
    print("This script contains the training function.")
    print("To run training, execute main.py.")