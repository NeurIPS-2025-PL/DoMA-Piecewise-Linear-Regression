import numpy as np
from Initialization import Initialize_MaX
from GradientStep import GradientStep


def Train_Max(k: int, M:int, X:np.ndarray, y:np.ndarray,G_steps:int  ) -> float:
    # Input Data:
    # X: Covariates \in R^ (d+1) x n
    # y: Measurements \in R^n
    # k: Number of linear segments
    # M: Number of random initializations
    
    # Initialization:
    Model_init,err_init = Initialize_MaX(k,M,y,X)
    print("Initial Error: ",err_init)

    # Alternating Block Gradient Descent (AB-GD):
    Model_hat,final_MSE, converged, total_steps = GradientStep(X,y,G_steps,k, Model_init)
    # print final MSE, convergence status, and total steps
    print(f"Final MSE: {final_MSE:.2e}, Total Steps: {total_steps}")

    return Model_hat, final_MSE, converged, total_steps