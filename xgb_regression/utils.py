import numpy as np

def get_target_transform(transform_name):
    """Get the target transformation function based on the specified name.
    Args:
        transform_name (str): Name of the transformation ('sqrt', 'log', 'none').
    Returns:
        function: A function that applies the specified transformation.
    """
    if transform_name == 'sqrt':
        return lambda x: np.sqrt(x)
    elif transform_name == 'log':
        return lambda x: np.log1p(x)
    elif transform_name == 'none':
        return lambda x: x
    else:
        raise ValueError(f"Unknown target_transform: {transform_name}")


def get_inverse_transform(transform_name):
    """Get the inverse target transformation function based on the specified name.
    Args:
        transform_name (str): Name of the transformation ('sqrt', 'log', 'none').
    Returns:
        function: A function that applies the specified inverse transformation.
    """
    if transform_name == 'sqrt':
        return lambda x: np.asarray(x, dtype=np.float64) ** 2
    elif transform_name == 'log':
        return lambda x: np.expm1(np.asarray(x, dtype=np.float64))
    elif transform_name == 'none':
        return lambda x: np.asarray(x, dtype=np.float64)
    else:
        raise ValueError(f"Unknown target_transform: {transform_name}")
