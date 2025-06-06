import torch
import numpy as np
from sklearn.decomposition import PCA
import joblib
import os

def get_target_transform(transform_name):
    if transform_name == 'sqrt':
        return lambda x: torch.sqrt(x)
    elif transform_name == 'log':
        return lambda x: torch.log1p(x)
    elif transform_name == 'none':
        return lambda x: x
    else:
        raise ValueError(f"Unknown target_transform: {transform_name}")

def get_inverse_transform(transform_name):
    if transform_name == 'sqrt':
        return lambda x: np.asarray(x, dtype=np.float64) ** 2
    elif transform_name == 'log':
        return lambda x: np.expm1(np.asarray(x, dtype=np.float64))
    elif transform_name == 'none':
        return lambda x: np.asarray(x, dtype=np.float64)
    else:
        raise ValueError(f"Unknown target_transform: {transform_name}")

def perform_pca(features: np.ndarray, n_components, pca_model_path: str=None, random_state: int=42) -> np.ndarray:
    """
    Fit PCA on features, transform them, and save the PCA model.

    Args:
        features (np.ndarray): Feature matrix.
        n_components (int): Number of PCA components.
        pca_model_path (str): Path to save the PCA model.
        random_state (int): Random state for reproducibility.

    Returns:
        np.ndarray: Transformed principal components.
    """
    # Check if the model exists and load it if available or fit a new one
    if pca_model_path is not None and os.path.exists(pca_model_path):
        # Load the existing PCA model
        print(f"PCA model already exists at {pca_model_path}. Loading the model.")
        pca = joblib.load(pca_model_path)
    else:
        # Fit a new PCA model
        print("Fitting PCA model...")
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(features)

        # Save the PCA model if a path is provided
        joblib.dump(pca, pca_model_path)
        print(f"PCA model saved to {pca_model_path}.")

    # Transform the features using the fitted PCA model
    features_pca = pca.transform(features)

    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    total_explained_variance = np.sum(explained_variance)
    print(f"Total explained variance by {n_components} components: {total_explained_variance:.6f}")
    print(f"Explained variance by PCA components: {np.round(explained_variance, 6)}")

    return features_pca
