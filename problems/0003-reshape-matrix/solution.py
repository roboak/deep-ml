import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	a = np.asarray(a)
	if sum(a.shape) != sum(new_shape):
		return []
	reshaped_matrix = np.reshape(a, new_shape).tolist()

	return reshaped_matrix