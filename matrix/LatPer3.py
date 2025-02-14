from numpy import array, zeros

print("\nNAMA : RIYAN WARDHANA")
print("NIM : 4233550008")

# LATIHAN SOAL NOOMR 1 TRANSPOSE MATRIX A
A = array([[1, 3, -6, -2],
           [5, 9, 7, 5.6],
           [2, 4, 8, -1],
           [2.3, 1.4, 0.8, -2.3]])
AT = zeros((4, 4))  
for i in range(0, 4):
    for j in range(0, 4):
        AT[i, j] = A[j, i]
print("TRANSPOSE MATRIX = A:\n", AT)


# LATIHAN SOAL NOOMR 1 TRANSPOSE MATRIX B
B = array([[8, 1, 4, 21],
           [3, 10, 5, 0.1],
           [7, -2, 9, -5],
           [2.7, -12, -8.9, 5.7]])
BT = zeros((4, 4))  
for i in range(0, 4):
    for j in range(0, 4):
        BT[i, j] = B[j, i]
print("TRANSPOSE MATRIX = B:\n", BT)

# LATIHAN SOAL NOOMR 1 TRANSPOSE MATRIX X
x = array([[0.4178],
           [-2.9587],
           [56.3069],
           [8.1]])
xT = zeros((1, 4))
for i in range(0, 4):
    xT[0, i] = x[i, 0]

print("TRANSPOSE MATRIX = X:\n", xT)


# LATIHAN SOAL NOMOR 3
C = zeros((4, 4)) 
for i in range(0, 4):
    for j in range(0, 4):
        C[i, j] = A[i, j] + B[i, j]
print("PENJUMLAHAN MATRIX A DAN B:\n", C)

# LATIHAN SOAL NOMOR 4
E = zeros((4,4))
for i in range(0, 4):
    for j in range(0, 4):
        for k in range(0, 4):
            E[i, j] = E[i, j] + A[i, k] * B[k, j]
print("PERKALIAN MATRIX A DAN B:\n", E)

# LATIHAN SOAL NOMOR 5
y = zeros((4, 1))
for i in range(0, 4):
    for k in range(0, 4):
        y[i, 0] = y[i, 0] + A[i, k] * x[k, 0]
print("PERKALIAN MATRIX A DENGAN VEKTOR X:\n", y)

# LATIHAN SOAL NOMOR 6
y = zeros((4, 1)) 
for i in range(0, 4):
    for k in range(0, 4):
        y[i, 0] = y[i, 0] + B[i, k] * x[k, 0]
print("PERKALIAN MATRIX B DENGAN VEKTOR X:\n", y)