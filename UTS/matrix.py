import numpy as np
print("================== ")
print("= RIYAN WARDHANA = ")
print("= 4233550008     = ")
print("================== ")
def eliminasi_gauss(A, b):
    n = len(A)
    C = np.zeros((n, n+1))
    for i in range(n):
        for j in range(n):
            C[i][j] = A[i][j]
        C[i][n] = b[i]
    
    for k in range(n-1):
        if C[k][k] == 0:
            for s in range(n+1):
                v = C[k][s]
                u = C[k+1][s]
                C[k][s] = u
                C[k+1][s] = v
        for i in range(k+1, n):
            m = C[i][k] / C[k][k]
            for j in range(k, n+1):
                C[i][j] = C[i][j] - m * C[k][j]
    
    X = np.zeros(n)
    X[n-1] = C[n-1][n] / C[n-1][n-1]
    for j in range(n-2, -1, -1):
        S = 0
        for i in range(j+1, n):
            S = S + C[j][i] * X[i]
        X[j] = (C[j][n] - S) / C[j][j]
    return X

A = np.array([[1, 1, 2],
              [2, 4, -3],
              [3, 6, -5]])

b = np.array([9, 1, 0])

solusi = eliminasi_gauss(A, b)
print("= SOLUSI PERSAMAAN LINEAR =")
print(f"x = {solusi[0]}")
print(f"y = {solusi[1]}")
print(f"z = {solusi[2]}")