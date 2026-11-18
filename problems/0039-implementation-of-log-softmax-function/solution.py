import numpy as np

def softmax(x):
    bias = np.max(x, axis = -1)
    y = np.exp(x - bias)
    return y / y.sum()
def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Your code here
    return np.log(softmax(scores))