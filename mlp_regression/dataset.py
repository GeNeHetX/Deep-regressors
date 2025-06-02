import torch
import pandas as pd
from torch.utils.data import Dataset

class MALDI_multisamples(Dataset):
    """
    Custom Dataset for MALDI data.
    Reads a peaks file and a pixels file, separates features and target, and converts them to PyTorch tensors.
    """
    def __init__(self, features: str, targets: str, target: str, excluded_slides: list = None):
        """Initialize the dataset by loading features and targets data.

        Args:
            features (str): Path to the features data file.
            targets (str): Path to the targets data file.
            target (str): Name of the target variable.

        Raises:
            ValueError: If the number of samples in features and targets do not match.
        """
        # Load the features and targets data
        self.features = pd.read_pickle(features)
        self.targets = pd.read_pickle(targets)

        # Check if the features and targets data are of the same length
        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError("Number of samples in features and targets do not match.")

        # Check if the target exists in the targets data
        if target not in self.targets.columns:
            raise ValueError(f"Target '{target}' not found in targets data.")
        
        # Clean the data by dropping bad samples
        if excluded_slides:
            self.features = self.features[~self.targets['run'].isin(excluded_slides)]
            self.targets = self.targets[~self.targets['run'].isin(excluded_slides)]

        # Ensure features and targets are aligned
        self.features = self.features.reset_index(drop=True)
        self.targets = self.targets.reset_index(drop=True)

        # Convert features and targets to PyTorch tensors
        self.features = torch.tensor(data=self.features.values, dtype=torch.float32)
        self.targets = torch.tensor(data=self.targets[target].values, dtype=torch.float32).unsqueeze(1)

        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError("Number of samples in features and targets do not match.")

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
        return self.features[index], self.targets[index]

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':

    # --- Configuration ---
    PATH = "data/MALDI_IHC/correlations/"
    PEAKS_PATH = f"{PATH}peaks_standardized_lasso.pkl"
    PIXELS_PATH = f"{PATH}pixels_filtered_lasso.pkl"
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