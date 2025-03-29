import numpy as np
import math

def matxvek(A, x):
    """Perkalian matriks A dengan vektor x"""
    return np.dot(A, x)

def matxmat(A, B):
    """Perkalian dua matriks A dan B"""
    return np.dot(A, B)

def matplusmat(A, B):
    """Penjumlahan dua matriks A dan B"""
    return np.add(A, B)

def rumusabc(a, b, c):
    """Menghitung akar-akar dari persamaan kuadrat ax^2 + bx + c"""
    D = b**2 - 4*a*c
    if D > 0:
        x1 = (-b + math.sqrt(D)) / (2*a)
        x2 = (-b - math.sqrt(D)) / (2*a)
        return x1, x2
    elif D == 0:
        x = -b / (2*a)
        return x, x
    else:
        x1 = complex(-b, math.sqrt(-D)) / (2*a)
        x2 = complex(-b, -math.sqrt(-D)) / (2*a)
        return x1, x2

def formula(a, b, c):
    """Menghitung dua hasil dengan formula y = a^2 + 2*b + c dan x = a*b*c"""
    y = a**2 + 2*b + c
    x = a * b * c
    return y, x

A = np.array([[3., 8., 5.], [6., 4., 7.]])
B = np.array([[2., 1., 3.], [5., 7., 6.]])
x = np.array([[2.], [3.], [4.]])

result_sum = matplusmat(A, B)
print("Penjumlahan Matriks A dan B:\n", result_sum)

result_mul = matxmat(A, B.T)  
print("Perkalian Matriks A dan B:\n", result_mul)

result_matxvek_A = matxvek(A, x)
print("Perkalian Matriks A dengan Vektor x:\n", result_matxvek_A)

result_matxvek_B = matxvek(B, x)
print("Perkalian Matriks B dengan Vektor x:\n", result_matxvek_B)

a, b, c = 1, -4, 4  
akar = rumusabc(a, b, c)
print("Akar Persamaan Kuadrat:", akar)

hasil_formula = formula(2, 3, 4)
print("Hasil Formula:", hasil_formula)
