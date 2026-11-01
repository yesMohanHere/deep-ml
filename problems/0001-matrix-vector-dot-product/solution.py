import torch


def matrix_dot_vector(a, b) -> torch.Tensor:
    """Compute the product of matrix `a` and vector `b` using PyTorch.

    Inputs can be Python lists, NumPy arrays, or torch Tensors. Returns a 1-D
    tensor of length m, or tensor(-1) if dimensions mismatch.
    """
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)

    # Ensure `a_t` is 2-D, `b_t` is 1-D, and inner dimensions match
    if a_t.ndim != 2 or b_t.ndim != 1 or a_t.size(1) != b_t.size(0):
        return torch.tensor(-1)

    # torch.mv computes the matrix-vector product
    return torch.mv(a_t, b_t)