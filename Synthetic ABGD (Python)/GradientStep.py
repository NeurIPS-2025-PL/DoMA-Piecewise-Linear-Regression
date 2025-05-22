import numpy as np
from Tropical_Functions import DMax_Function


def GradientStep(X:np.ndarray,y:np.ndarray,G_steps:int,k:int, Model_init:np.ndarray)-> tuple[np.ndarray, float,bool,int]:
    n = X.shape[1]
    mu = 1
    New_Model = Model_init
    verbose = True
    converged = False
    for i in range(G_steps):
         Old_Model = New_Model
         yhat, Ctop, Cbot = DMax_Function(X,Old_Model)
         err = yhat - y
         Xe = X * err.T
         Top_match = np.zeros((n,2*k))
         Top_match[np.arange(n),Ctop[:,0]] = 1
         Bot_match = np.zeros((n,2*k))
         Bot_match[np.arange(n),(Cbot[:,0]+k)] = 1
         for j in range(2*k):
             njtop = np.sum(Top_match[:,j])
             njbot = np.sum(Bot_match[:,j])
             if njtop != 0:
                    Top_match[:,j] = Top_match[:,j]*n/njtop 
             if njbot != 0:
                    Bot_match[:,j] = Bot_match[:,j]*n/njbot

         if i%2 == 0:
               gradient  = Xe @ Top_match /n/2
         else:
               gradient  = -Xe @ Bot_match /n/2
            
         New_Model = Old_Model - mu * gradient
         # Check Convergence:
         decay = np.linalg.norm(New_Model - Old_Model)/np.linalg.norm(Old_Model)
         MSE = np.linalg.norm(err)**2/n
         if decay > 1e-8:
               if (i+1)%500 == 0 and verbose:
                    print(f"Step {i+1}: Decay = {decay:.2e}, MSE = {MSE:.2e}")
         else:
             if verbose:
                    print(f"Step {i+1}: Decay = {decay:.2e}, MSE = {MSE:.2e}")
                    print("Converged!")
             converged = True
             break
    return New_Model, MSE, converged, (i+1)