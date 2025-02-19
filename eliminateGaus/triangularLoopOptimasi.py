from numpy import array, zeros
A = array([[1.,1.,0.,3.,4],\
           [2.,1.,-1.,1.,1],\
            [3.,-1.,-1.,2.,-3],\
                        [-1.,2.,3.,-1,4]])
print (A)

for k in range(0,3):
            for j in range(k+1,4):
                    m=A[j,k]/A[k, k]
                    for i in range(0,5):
                            A[j, i]=A[j, i]-m*A[k, i]

print (A)

X = zeros((4,1))
X[3,0]=A[3,4]/A[3,3]
X[2,0]=(A[2,4]-A[2,3]*X[3,0])/A[2,2]
X[1,0]=(A[1,4]-(A[1,2]*X[2,0]+A[1,3]*X[3,0]))/A[1,1]
X[0,0]=(A[0,4]-(A[0,1]*X[1,0]+A[0,2]*X[2,0]+A[0,3]*X[3,0]))/A[0,0]

print (X)