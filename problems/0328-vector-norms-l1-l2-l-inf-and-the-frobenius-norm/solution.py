import numpy as np


def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    arr = np.asarray(arr)

    if norm_type == "frobenius":
        if arr.ndim != 2:
            raise ValueError(f"Frobenius norm requires a 2D array, got {arr.ndim}D.")
        return float(np.linalg.norm(arr, ord="fro"))

    if arr.ndim not in (1, 2):
        raise ValueError(f"Entrywise norms require a 1D or 2D array, got {arr.ndim}D.")

    if norm_type == "l1":
        return float(np.linalg.norm(arr.ravel(), ord=1))

    if norm_type == "l2":
        return float(np.linalg.norm(arr.ravel(), ord=2))

    if norm_type == "linf":
        return float(np.linalg.norm(arr.ravel(), ord=np.inf))

    raise ValueError(
        f"Invalid norm_type '{norm_type}'. Expected 'l1', 'l2', 'linf', or 'frobenius'."
    )