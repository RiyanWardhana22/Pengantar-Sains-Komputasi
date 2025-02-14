from __future__ import division
from numpy import sqrt, complex

def rumusabc(a,b,c):
            D = b*b-4*a*c
            if D > 0.0:
                    x1 = (-b+sqrt(D))/(2*a)
                    x2 = (-b-sqrt(D))/(2*a)
            elif D == 0.0:
                    x1 = -b/(2*a)
                    x2 = -b/(2*a)
            else:
                    D = -D
                    x1r = -b/(2*a)
                    x1i = sqrt(D)/(2*a)
                    x1 = complex(x1r,x1i)
                    x2r = x1r
                    x2 = complex()