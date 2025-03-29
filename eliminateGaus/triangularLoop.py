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

# MENGHILANGKAN x1 dari p3 dan p4
m=A[2,1]/A[1,1]
for i in range(0,5):
            A[2, i]=A[2, i]-m*A[1, i]

m=A[3,1]/A[1,1]
for i in range(0,5):
            A[3, i]=A[3, i]-m*A[1, i]

m=A[3,2]/A[2,2]
for i in range(0,5):
            A[3, i]=A[3, i]-m*A[2, i]

print (A)
X = zeros((4,1))
X[3,0]=A[3,4]/A[3,3]
X[2,0]=(A[2,4]-A[2,3]*X[3,0])/A[2,2]
X[1,0]=(A[1,4]-(A[1,2]*X[2,0]+A[1,3]*X[3,0]))/A[1,1]
X[0,0]=(A[0,4]-(A[0,1]*X[1,0]+A[0,2]*X[2,0]+A[0,3]*X[3,0]))/A[0,0]

print (X)
