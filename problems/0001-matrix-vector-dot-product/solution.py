def matrix_dot_vector(
    a: list[list[int | float]], b: list[int | float]
) -> list[int | float] | int:
    """Return a list where each element is the dot product of a row of 'a' with 'b'.

    If the number of columns in 'a' does not match the length of 'b', return -1.
    """
    # Verify non-empty matrix and matching dimensions
    if not a or len(a[0]) != len(b):
        return -1

    # Ensure every row matches the vector length (guards against ragged matrices)
    if any(len(row) != len(b) for row in a):
        return -1

    # Compute dot product of each row with b
    return [sum(x * y for x, y in zip(row, b)) for row in a]