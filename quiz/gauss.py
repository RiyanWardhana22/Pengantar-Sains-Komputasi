import numpy as np
A = np.array([
    [4, 6, 0],  
    [0, 6, 2],  
    [4, 0, 2]   
])

B = np.array([14, 10, 14])
I = np.linalg.solve(A, B)

print(f"Arus i1 = {I[0]} A")
print(f"Arus i2 = {I[1]} A")
print(f"Arus i3 = {I[2]} A")

