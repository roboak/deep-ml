def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    n_rows, n_cols = len(a), len(a[0])
    n_rows_t, n_cols_t = n_cols, n_rows
    # a_t = [[None]*n_cols_t]*n_rows_t
    a_t = [[None] * n_cols_t for _ in range(n_rows_t)]

    # print(a_t)
    for i in range(n_rows_t):
        for j in range(n_cols_t):
            # print (i, j)
            a_t[i][j] = a[j][i]
            # print(a_t)
    
    return a_t