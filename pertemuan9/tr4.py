import numpy as np
print("\nNAMA : RIYAN WARDHANA")
print("NIM  : 4233550008")
print("=====================")

def eleminasi_gauss(A, b):
    n = len(b)
    ditambah_matrix = np.hstack((A, b.reshape(n, 1)))
    for i in range(n):
        ditambah_matrix[i] = ditambah_matrix[i] / ditambah_matrix[i, i]
        for j in range(i + 1, n):
            ditambah_matrix[j] = ditambah_matrix[j] - ditambah_matrix[j, i] * ditambah_matrix[i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = ditambah_matrix[i, -1] - np.dot(ditambah_matrix[i, i+1:n], x[i+1:n])
    return x

G = 6.67430e-11 
M = 5.972e24     
x, y, z = 7e6, 7e6, 7e6  
r = np.sqrt(x**2 + y**2 + z**2)

A = np.array([
    [2*x, y, z],
    [x, 3*y, z],
    [x, y, 4*z]
])

b = np.array([
    -G * M / r**2 * x**2,
    -G * M / r**2 * y**2,
    -G * M / r**2 * z**2
])

a = eleminasi_gauss(A, b)
print("PERCEPATAN a_x:", a[0])
print("PERCEPATAN a_y:", a[1])
print("PERCEPATAN a_z:", a[2])