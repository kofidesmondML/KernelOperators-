import numpy as np

def test(x):
    n=len(x)
    s=1
    for i in range(n):
        s=s*x[i]*(1-x[i])
    return 4**n*(s)

def frankes(x,y):
    s1=(3/4)*np.exp(-1/4*((9*x-2)**2+(9*y-2)**2))
    s2=(3/4)+np.exp(-1/49*(9*x+1)**2-1/10*(9*y+1))
    s3=1/2*np.exp(-1/4*((9*x-7)**2+(9*y-3)**2))
    s4=(1/5)*np.exp(-1*(9*x-4)**2 -1*(9*y-7)**2 )
    
    return s1+s2+s3-s4    