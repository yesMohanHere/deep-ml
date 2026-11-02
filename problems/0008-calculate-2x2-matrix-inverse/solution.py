import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    try:
        # np.linalg.inv automatically calculates the inverse
        inv_matrix = np.linalg.inv(matrix)
        
        # Convert the resulting NumPy array back to a list of lists
        return inv_matrix.tolist()
        
    except np.linalg.LinAlgError:
        # NumPy raises a LinAlgError if the matrix is singular (determinant is 0)
        return None