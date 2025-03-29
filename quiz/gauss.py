import numpy as np
A = np.array([
    [4, 6, 0],  
    [0, 6, 2],  
    [4, 0, 2]   
])

B = np.array([14, 10, 14])
I = np.linalg.solve(A, B)

print(f"ARUS i1 = {I[0]} A")
print(f"ARUS i2 = {I[1]} A")
print(f"ARUS i3 = {I[2]} A")


