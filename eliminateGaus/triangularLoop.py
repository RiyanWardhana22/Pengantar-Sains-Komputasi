from numpy import array, zeros
A = array([[1.,1.,0.,3.,4],\
           [2.,1.,-1.,1.,1],\
            [3.,-1.,-1.,2.,-3],\
                        [-1.,2.,3.,-1,4]])

print (A)
m=A[1,0]/A[0,0]
for i in range(0,5):
            A[1, i]=A[1, i]-m*A[0, i]

m=A[2,0]/A[0,0]
for i in range(0,5):
            A[2, i]=A[2, i]-m*A[0, i]

m=A[3,0]/A[0,0]
for i in range(0,5):
            A[3, i]=A[3, i]-m*A[0, i]
