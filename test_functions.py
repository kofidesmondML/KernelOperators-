import numpy as np

def test_function(x):
    n = x.shape[1] 
    s = np.ones(x.shape[0]) 
    for i in range(x.shape[0]):  
        product = 1
        for j in range(n):  
            product *= x[i, j] * (1 - x[i, j])
        s[i] = 4**n * product 
    return s

def frankes(x,y):
    s1=(3/4)*np.exp(-1/4*((9*x-2)**2+(9*y-2)**2))
    s2=(3/4)+np.exp(-1/49*(9*x+1)**2-1/10*(9*y+1))
    s3=1/2*np.exp(-1/4*((9*x-7)**2+(9*y-3)**2))
    s4=(1/5)*np.exp(-1*(9*x-4)**2 -1*(9*y-7)**2 )
    
    return s1+s2+s3-s4    