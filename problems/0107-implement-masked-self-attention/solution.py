import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(x):
    v = np.exp(x - np.max(x, axis=1, keepdims=True))
    return v/np.sum(v, axis = 1, keepdims=True)


def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    # Your code here
    # print(K@Q.T/np.sqrt(K.shape[1]) + mask)
    return softmax(Q@K.T/np.sqrt(K.shape[1]) + mask)@V
