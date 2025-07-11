import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
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


def perform_pca(features: np.ndarray, n_components, model_base_path: str=None, random_state: int=42) -> np.ndarray:
    """
    Fit PCA on features, transform them, and save the PCA model.

    Args:
        features (np.ndarray): Feature matrix.
        n_components (int): Number of PCA components.
        model_base_path (str): Base path to save the PCA model.
        random_state (int): Random state for reproducibility.

    Returns:
        np.ndarray: Transformed principal components.
    """
    model_path = f"{model_base_path}_pca.joblib"
    # Check if the model exists and load it if available or fit a new one
    if os.path.exists(model_path):
        # Load the existing PCA model
        print(f"PCA model already exists at {model_path}. Loading the model.")
        pca = joblib.load(model_path)
    else:
        # Fit a new PCA model
        print("Fitting PCA model...")
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(features)

        # Save the PCA model
        joblib.dump(pca, model_path)
        print(f"PCA model saved to {model_path}.")

    # Transform the features using the PCA model
    features_pca = pca.transform(features)

    # Calculate explained variance ratio
    total_explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Total explained variance by {n_components} components: {total_explained_variance:.4f}")

    return features_pca


def perform_ica(
    features: np.ndarray,
    n_components,
    model_base_path: str,
    random_state: int = 42
) -> np.ndarray:
    """
    Always perform PCA before ICA, save both models using a common base path.

    Args:
        features (np.ndarray): Feature matrix.
        n_components (int): Number of components for both PCA and ICA.
        model_base_path (str): Base path for saving/loading models (without extension).
        random_state (int): Random state for reproducibility.

    Returns:
        np.ndarray: Transformed independent components.
    """

    print(f"Applying PCA to reduce features to {n_components} components before ICA...")
    features_pca = perform_pca(features, n_components, model_base_path, random_state)

    ica_model_path = f"{model_base_path}_ica.joblib"
    # Check if the ICA model exists and load it if available or fit a new one
    if os.path.exists(ica_model_path):
        print(f"ICA model already exists at {ica_model_path}. Loading the model.")
        ica = joblib.load(ica_model_path)
    else:
        # Fit a new ICA model
        print("Fitting ICA model...")
        ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
        ica.fit(features_pca)

        # Save the ICA model
        joblib.dump(ica, ica_model_path)
        print(f"ICA model saved to {ica_model_path}.")

    # Transform the features using the ICA model
    features_ica = ica.transform(features_pca)
    
    return features_ica


def perform_svd(
    features: np.ndarray,
    n_components: int,
    model_base_path: str,
    random_state: int = 42
) -> np.ndarray:
    """
    Perform SVD on the features and save the model.

    Args:
        features (np.ndarray): Feature matrix.
        n_components (int): Number of components for SVD.
        model_base_path (str): Base path for saving/loading models (without extension).
        random_state (int): Random state for reproducibility.

    Returns:
        np.ndarray: Transformed features after SVD.
    """
    svd_model_path = f"{model_base_path}_svd.joblib"
    if os.path.exists(svd_model_path):
        print(f"SVD model already exists at {svd_model_path}. Loading the model.")
        svd = joblib.load(svd_model_path)
    else:
        print("Fitting SVD model...")
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        svd.fit(features)

        # Inspect the explained variance
        explained_variance = svd.explained_variance_ratio_.sum()
        print(f"Total explained variance by {n_components} components: {explained_variance:.4f}")

        joblib.dump(svd, svd_model_path)
        print(f"SVD model saved to {svd_model_path}.")

    features_svd = svd.transform(features)
    
    return features_svd

def perform_dim_reduction(
    features: np.ndarray,
    n_components: int,
    model_base_path: str,
    method: str = "pca",
    random_state: int = 42
) -> np.ndarray:
    """
    Perform dimensionality reduction using PCA or ICA (always with PCA first for ICA).

    Args:
        features (np.ndarray): Feature matrix.
        n_components (int): Number of components for both PCA and ICA.
        model_base_path (str): Base path for saving/loading models (without extension).
        method (str): 'pca' or 'ica'.
        random_state (int): Random state for reproducibility.

    Returns:
        np.ndarray: Transformed features.
    """
    if method == "pca":
        return perform_pca(features, n_components, model_base_path, random_state)
    elif method == "ica":
        return perform_ica(features, n_components, model_base_path, random_state)
    elif method == "svd":
        return perform_svd(csr_matrix(features), n_components, model_base_path, random_state)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
