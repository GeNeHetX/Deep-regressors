import torch
from torch.utils.data import Dataset

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

    # --- Configuration ---
    PATH = "data/MALDI_IHC/correlations/"
    PEAKS_PATH = f"{PATH}peaks_standardized_lesion.pkl"
    PIXELS_PATH = f"{PATH}pixels_filtered_lesion.pkl"
    TARGET = 'Density_CD8'

    # Example: Load the dataset
    dataset = MALDI_multisamples(peaks=PEAKS_PATH, pixels=PIXELS_PATH, target=TARGET)
    print("Dataset loaded.")
    print(f"Number of samples: {len(dataset)}")
    print(f"Number of features: {dataset.n_features}")

    # Example: Get the first sample
    first_features, first_target = dataset[0]
    print(f"First sample features: {first_features}")
    print(f"First sample target: {first_target}")