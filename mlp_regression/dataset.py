import torch
from torch.utils.data import Dataset
import numpy as np

class TableDataset(Dataset):
    """
    Custom Dataset for tabular data.
    Accepts features and target arrays, applies optional transformation.
    """
    def __init__(self, features, target, target_transform=None):
        """
        Args:
            features (array-like): Features data (numpy array or DataFrame.values).
            target (array-like): Target vector (numpy array or Series).
            target_transform (callable, optional): A function to transform the target variable.
        """
        # Check if features and target are of the same length
        if features.shape[0] != target.shape[0]:
            raise ValueError("Number of samples in features and target do not match.")

        # Initialize the features and target as PyTorch tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

        # Apply target transformation if provided
        if target_transform is not None:
            self.target = target_transform(self.target)

        # Ensure target is a 2D tensor with shape (n_samples, 1)
        self.target = self.target.unsqueeze(1)

        # Store the number of samples and features
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]

    def __len__(self):
        """
        Return the total number of samples

        Returns:
            int: Number of samples in the dataset.
        """
        # Return the total number of samples
        return self.n_samples

    def __getitem__(self, index):
        """
        Return one sample (features and target) at the given index.
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the features and target for the sample.
        """
        # Return one sample (features and target) at the given index
        return self.features[index], self.target[index]

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':

    # Create a small dataset for testing
    features = np.random.rand(10, 5)  # 10 samples, 5 features
    target = np.random.rand(10)        # 10 target values

    # Create the dataset
    dataset = TableDataset(features, target)

    # Print dataset length and first item
    print(f"Dataset length: {len(dataset)}")
    print("First item (features, target):", dataset[0])