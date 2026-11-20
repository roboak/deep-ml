import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    B, C, H, W = X.shape
    mini_C = C // num_groups
    updated_X = X.reshape((B, num_groups, mini_C, H, W))
    avg = np.mean(updated_X, axis = (2, 3, 4), keepdims = True) # (1, n_groups, 1, 1, 1)
    var = np.var(updated_X, axis = (2, 3, 4), keepdims = True) # (1, n_groups, 1, 1, 1)
    
    # B, n_groups, mini_C, 1, 1, 1
    norm_X = (updated_X - avg )/np.sqrt(var + epsilon)
    norm_X = norm_X.reshape((B, C, H, W))
    return gamma*norm_X + beta

