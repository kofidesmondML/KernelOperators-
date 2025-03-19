import numpy as np 


def distance_matrix(dsites, ctrs):
    (m,s)=dsites.shape
    print(dsites.shape)
    (n,s)=ctrs.shape
    dm=np.zeros((m,n))
    for d in range(s):
        dr, cc = np.meshgrid(dsites[:, d], ctrs[:, d], indexing='ij')
        dm += (dr - cc) ** 2  
    return np.sqrt(dm)
        
    
    
    