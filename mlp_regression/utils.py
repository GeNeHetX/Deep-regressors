import torch
import numpy as np

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
