import numpy as np
def gauss(A, b):
    n = len(A)
    Ab = np.hstack([A, b])
    for i in range(n):
        if Ab[i, i] == 0:
            for k in range(i + 1, n):
                if Ab[k, i] != 0:
                    Ab[[i, k]] = Ab[[k, i]]
                    break
        Ab[i] = Ab[i] / Ab[i, i]
        for j in range(i + 1, n):
            Ab[j] -= Ab[j, i] * Ab[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.sum(Ab[i, i+1:n] * x[i+1:n])
    return x
A = np.array([[1., 1., -1.],
              [6., -4., 0.],
              [6., 0., 2.]])

b = np.array([[0.],
              [24.],
              [10.]])

arus = gauss(A, b)
print("ARUS I1, I2, I3:", arus)
