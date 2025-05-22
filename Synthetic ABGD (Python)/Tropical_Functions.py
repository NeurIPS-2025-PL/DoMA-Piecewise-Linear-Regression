import numpy as np

# Max Affine Function:
def Max_Function(X,Model,sig=0)-> tuple[np.ndarray, np.ndarray]:
    # Input:
    # X:        Covariates \in R^ (d+1) x n
    # Model:    Model Parameters \in R^ (d+1) x k
    # sig:      Noise standard Deviation

    #Output:    Max-affine function evaluation and index of the linear segment
    yj = X.T @ Model
    y = yj.max(axis = 1) + np.random.normal(loc=0,scale=sig,size=(X.shape[1],))
    Cj = yj.argmax(axis = 1)
    return y[:,None], Cj[:,None]

# DMax Affine Function:
def DMax_Function(X,Model,sig=0)-> tuple[np.ndarray, np.ndarray]:
    # Input:
    # X:        Covariates \in R^ (d+1) x n
    # Model:    Model Parameters \in R^ (d+1) x 2k
    # sig:      Noise standard Deviation

    #Output:    Max-affine function evaluation and index of the linear segment
    k = Model.shape[1]//2
    yj = X.T @ Model[:,:k]
    ytop = yj.max(axis = 1) 
    Ctop = yj.argmax(axis = 1)
    yj = X.T @ Model[:,k:]
    ybot = yj.max(axis = 1) 
    Cbot = yj.argmax(axis = 1)
    y = ytop - ybot + np.random.normal(loc=0,scale=sig,size=(X.shape[1],))
    return y[:,None], Ctop[:,None], Cbot[:,None]
