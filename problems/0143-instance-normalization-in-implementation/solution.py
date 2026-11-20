import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    # TODO: Implement Instance Normalization
    avg = np.mean(X, axis = (2, 3), keepdims=True)
    var = np.var(X, axis = (2, 3), keepdims=True)
    norm_X = (X-avg)/np.sqrt(var + epsilon)
    return gamma*norm_X + beta