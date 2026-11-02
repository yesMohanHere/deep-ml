import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	# Your code here
	if not matrix:
		return []
	return (np.array(matrix)*scalar).tolist()