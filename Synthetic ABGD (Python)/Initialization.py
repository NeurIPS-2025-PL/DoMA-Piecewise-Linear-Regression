import numpy as np
from Tropical_Functions import DMax_Function



def Initialize_MaX(k:int,M:int,y:np.ndarray,X =np.ndarray)-> tuple[np.ndarray, float]:
    #Input:
    #y:     n x 1 Max-affine function evaluation
    #X:     (d+1) x n Covariates (Last row is 1's)
    #k:     Number of linear segments
    d = X.shape[0]-1
    n = X.shape[1]

    # Subspace Estimation:
    #if 2*k>= d:
    #    U = np.identity(d)          # Full Rank
    #else:
    #    U = PCA_method(y,X,k,d,n,TrueModel)   # Principal Component Analysis
    ns  = min(d,2*k)
    U = np.identity(d)[:,:ns] 
    
    # Model Initialization
    ecand = np.inf
    for i in range(M):
        kcandidate = np.random.normal(loc=0.0, scale=1, size=(ns,2*k)) # Low Dim Random Initialization
        bcandidate = U @ kcandidate # Projection to High Dim
        ccandidate = np.random.normal(loc=0.0, scale=1, size=(1,2*k)) # Random Bias
        tcandidate = np.vstack((bcandidate,ccandidate)) # Model Initialization
        #tcandidate[:,-1] = 0
        #tcandidate = GradientStep(x,y,tcandidate) # Gradient Descent
        yhat, _, _ =  DMax_Function(X,tcandidate)
        ehat = np.linalg.norm(y-yhat)**2/n *100
        if ehat<ecand:
            ecand = ehat
            Model = tcandidate

    return Model, ecand

'''
def PCA_method(y,X,k,d,n,TrueModel):
    Xtop = X[:-1,:]
    tmp = np.sum(Xtop*y,axis=1)/n
    tmp = tmp[:,None]
    M = tmp @ tmp.T + (Xtop@np.diag(y)@Xtop.T - np.sum(y)*np.identity(d))/n
    eigvalues, V = np.linalg.eigh(M)
    indk = np.flip(np.argsort(eigvalues))
    U = V[:,indk[:2*k]]
    return U
'''