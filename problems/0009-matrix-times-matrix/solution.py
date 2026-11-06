import numpy as np

def matrixmul(a: list[list[int|float]],
              b: list[list[int|float]]) -> list[list[float]] | int:
    
    # Catch empty lists to prevent IndexErrors
    if not a or not b:
        return -1

    # For matrix multiplication, columns in 'a' must equal rows in 'b'
    if len(a[0]) != len(b):
        return -1
    
    # np.dot returns a NumPy array, so we convert it back to a list
    return np.dot(a, b).tolist()