import numpy as np

def softmax(values):
    # Implement the softmax function
    v = np.exp(values - np.max(values))
    # print(np.sum(v, axis = 1).shape)
    # print(np.sum(v, axis = 1, keepdims = True).shape)
    return v / np.sum(v, axis = 1, keepdims = True)

def pattern_weaver(n, crystal_values, dimension):
    # Your code here
    crystal_np = np.array(crystal_values)[:, None]
    self_correation = (np.matmul(crystal_np, crystal_np.T))/ np.sqrt(dimension)

    out = softmax(self_correation) @ crystal_np
    
    return list(out.flatten())
