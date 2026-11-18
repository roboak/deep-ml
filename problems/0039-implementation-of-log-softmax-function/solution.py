import numpy as np

def softmax(x):
    y = np.exp(x)
    return y / y.sum()
def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Your code here
    return np.log(softmax(scores))