import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pandas as pd

class MSI_Image_Dataset(Dataset):
    def __init__(self, features: pd.DataFrame, coordinates: pd.DataFrame, samples_indices: np.ndarray, target: np.ndarray, transform=None, feature_transform=None, target_transform=None, img_size: tuple = (1024, 1024)):
        """
        features: DataFrame of features (n_samples, n_features)
        coordinates: DataFrame with 'x', 'y', and 'run' columns (n_samples, ...)
        samples_indices: numpy array indicating the slide/run each sample belongs to (n_samples,)
        target: numpy array, target values for the dataset
        transform: optional transform to apply to both feature and target images
        feature_transform: optional transform to apply to the feature images
        target_transform: optional transform to apply to the target images
        img_size: tuple indicating the target size for the images (height, width)
        """
        self.transform = transform
        self.feature_transform = feature_transform
        self.target_transform = target_transform
        self.target = target

        # Group indices by samples
        self.n_observations, self.n_features = features.shape
        self.samples_ids = np.unique(samples_indices)
        self.samples_indices = samples_indices
        self.features = features
        self.coordinates = coordinates

        # Calculate global max coordinates for all samples
        self.target_height, self.target_width = img_size

    def __len__(self):
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples_ids)

    def __getitem__(self, idx):
        """Get a single item from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image tensor, label tensor, padding sequence) where padding_sequence is a tuple of (left, right, top, bottom) padding integer values.
        """
        # Extract the sample features and pixels
        sample = self.samples_ids[idx]
        indices = self.samples_indices == sample
        sample_features = self.features.iloc[indices].values.astype('float32')
        sample_coordinates = self.coordinates.iloc[indices]

        # Extract the sample coordinates and labels
        x_coords, y_coords = sample_coordinates['x'].values.astype(int), sample_coordinates['y'].values.astype(int)
        labels = self.target[indices].astype('float32')

        # Get image dimensions for current sample
        channels = sample_features.shape[1]
        height = y_coords.max() + 1
        width = x_coords.max() + 1

        # Initialize zero images for the features and labels
        img = np.zeros((channels, height, width), dtype='float32')
        label_img = np.zeros((height, width), dtype='float32')

        # Fill the zero images with the corresponding values
        for i in range(len(sample_features)):
            img[:, y_coords[i], x_coords[i]] = sample_features[i]
            label_img[y_coords[i], x_coords[i]] = labels[i]

        # Transform the images arrays
        if self.transform:
            img , label_img = self.transform(img, label_img)

        # Convert to tensors
        img, label_img = torch.from_numpy(img), torch.from_numpy(label_img)

        # Add channel dimension to label image
        label_img = label_img.unsqueeze(0)

        # Apply any additional transformations
        if self.feature_transform:
            img = self.feature_transform(img)
        if self.target_transform:
            label_img = self.target_transform(label_img)

        # Pad images and labels to global target size
        pad_height = self.target_height - img.shape[1]
        pad_width = self.target_width - img.shape[2]
        if pad_height > 0 or pad_width > 0:
            # Compute padding for each side
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left

            # Create padding sequences
            padding_sequence = (pad_left, pad_right, pad_top, pad_bottom)

            # Apply padding to the features and labels images
            img = F.pad(img, padding_sequence)
            label_img = F.pad(label_img, padding_sequence)

        return img, label_img, padding_sequence

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':

    # Create a small dataset for testing
    features = pd.DataFrame(np.random.rand(10, 5))  # 10 samples, 5 features
    pixels = pd.DataFrame({
        'x': np.random.randint(0, 10, 10),
        'y': np.random.randint(0, 10, 10),
        'run': np.random.randint(0, 2, 10),
        'Density_CD8': np.random.rand(10)
    })

    dataset = MSI_Image_Dataset(features, samples_indices=pixels['run'].values, coordinates=pixels[['x', 'y']], target=pixels['Density_CD8'].values)

    # Print dataset length and first item
    print(f"Max coordinates: {dataset.max_coords}")
    print(f"Number of observations: {dataset.n_observations}")
    print(f"Number of features: {dataset.n_features}")
    print(f"Dataset length: {len(dataset)}")
    print("First item (image, label, slide):", dataset[0])