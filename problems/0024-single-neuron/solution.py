import math
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    # Your code here
    probabilities = sigmoid(np.array(features)  @ np.array(weights) + bias)
    mse = (np.sum(np.power(probabilities - np.array(labels), 2))) / len(features)
    return list(probabilities), mse