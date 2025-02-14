from numpy import zeros, transpose, sqrt, complex64, array
def matxvek(A, x):
    n = len(A)
    m = len(transpose(A))
    E = zeros((n, 1))
    for i in range(0, n):
        for k in range(0, m):
            E[i, 0] = E[i, 0] + A[i, k] * x[k, 0]
    return E

def matxmat(A, B):
    n = len(A)
    m = len(transpose(A))
    p = len(transpose(B))
    C = zeros((n, p))
    for i in range(0, n):
        for j in range(0, p):
            for k in range(0, m):
                C[i, j] = C[i, j] + A[i, k] * B[k, j]
    return C

def matplusmat(A, B):
    n = len(A)
    m = len(transpose(A))
    C = zeros((n, m))
    for i in range(0, n):
        for j in range(0, m):
            C[i, j] = A[i, j] + B[i, j]
    return C

def rumusabc(a, b, c):
    D = b*b - 4*a*c
    if D > 0.0:
        x1 = (-b + sqrt(D))/(2*a)
        x2 = (-b - sqrt(D))/(2*a)
    elif D == 0.0:
        x1 = -b/(2*a)
        x2 = -b/(2*a)
    else:
        D = -D
        x1r = -b/(2*a)
        x11 = sqrt(D)/(2*a)
        x1 = complex64(x1r, x11)
        x2r = x1r
        x21 = -x11
        x2 = complex64(x2r, x21)
    return x1, x2

def formula(a, b, c):
    y = a*a + 2*b*c
    x = a*b*c
    return y, x


A = array([[0.1, 2.3, -9.6, -2.7],
           [21.5, 2.9, 1.7, 5.6],
           [2.13, 4.29, 8.72, -1.02],
           [-2.3, 1.24, -0.18, 7.3]])

B = array([[8.3, 1.6, 4.8, 21.2],
           [3.4, 10.5, 5.2, 0.1],
           [7.8, -2.7, 9.4, -5.1],
           [2.7, -12.3, -18.9, 50.7]])

x = array([[23.78],
           [-7.97],
           [8.369],
           [4.112]])

print("\nNAMA : RIYAN WARDHANA")
print("NIM  : 42233550008")
print("=======================")

penjumlahan = matplusmat(A, B)
print("\nHASIL PENJUMLAHAN MATRIK A DAN B")
print(penjumlahan)

perkalian = matxmat(A, B)
print("\nHASIL PERKALIAN MATRIK A DAN B")
print(perkalian)

perkalian = matxvek(A, x)
print("\nHASIL PERKALIAN MATRIK A DAN VEKTOR X")
print(perkalian)

perkalian = matxvek(B, x)
print("\nHASIL PERKALIAN MATRIK B DAN VEKTOR X")
print(perkalian)