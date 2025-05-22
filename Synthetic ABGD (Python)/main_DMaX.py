# Imports:
import numpy as np
import matplotlib.pyplot as plt 
from Tropical_Functions import DMax_Function 
from Train_MaX import Train_Max

# Fix seed for reproducibility:
np.random.seed(1)

# Variable Setup:
d       =    int(1) #(1,2,25,200)      # Input Dimension (plotting is only available for d=1,2)
sig     =    1/100                     # Noise standard deviation
M       =    int(2e3)                  # Number of random initializations 
k       =    int(3)                    # Number of linear functions 
n       =    int(5e2)                  # Sample Size
plotme  =    True                      # Plot ON/OFF
G_steps =    int(5e3)                  # Number of Gradient Steps

# Generate Ground-Truth Model Parameters \in R^ (d+1) x k
Model = np.zeros(( d+1, 2*k))
Rand = np.random.normal(loc=0.0, scale=1, size=(d+1,2*k))
for jj in range(2*k):
    Model[:,jj] =  Rand[:,jj]/np.linalg.norm(Rand[:,jj])

# Generate Covariate Samples:
X = np.vstack((np.random.normal(loc=0.0, scale=1, size=(d,n)),np.ones((1,n)) )  )

# Generate Measurements:
y, _, _ = DMax_Function(X,Model,sig)

# Train Piecewise Linear Function:
Model_hat, final_MSE, converged, total_steps = Train_Max(k,M, X, y,G_steps)

# Generate Predictions:
yhat, _, _ = DMax_Function(X,Model_hat,0)

# Plot 
if plotme:
    if d ==1:
        # Plot Ground Truth and Prediction
        ind_x = np.argsort(X[0,:])
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({
            'text.usetex': True,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'figure.autolayout': True,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        plt.plot(X[0, :][ind_x], y[ind_x], '.',color = 'royalblue', label=r"$f(\cdot, \mathbf{\beta}, \mathbf{\alpha})$")
        plt.plot(X[0, :][ind_x], yhat[ind_x], 'r--', label=r"$f(\cdot, \hat{\mathbf{\beta}}, \hat{\mathbf{\alpha}})$")
        plt.legend()
        plt.xlabel(r"$x$", fontsize=12, labelpad=10)
        plt.ylabel(r"$y$", fontsize=12, labelpad=10)
        plt.title(r"Dataset and AB-GD Solution", fontsize=14)
        plt.legend(loc= 'best')
        plt.grid()
        plt.show()
    elif d==2:
        # Plot Ground Truth and Prediction
        X0 = X[0,:][:,None]
        X1 = X[1,:][:,None]
        fig = plt.figure(figsize=(8, 6))
        plt.rcParams.update({
            'text.usetex': True,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.dpi': 300,
            'figure.autolayout': True,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X0, X1, y, color='royalblue', marker='o', label=r"$f(\cdot, \mathbf{\beta}, \mathbf{\alpha})$")
        ax.plot_trisurf(X0.flatten(), X1.flatten(), yhat.flatten(), color='red', alpha=0.5, label=r"$f(\cdot, \hat{\mathbf{\beta}}, \hat{\mathbf{\alpha}})$")
        ax.set_xlabel(r"$x_1$", fontsize=12, labelpad=10)
        ax.set_ylabel(r"$x_2$", fontsize=12, labelpad=10)
        ax.set_zlabel(r"$y$", fontsize=12, labelpad=10)
        plt.title(r"Dataset and AB-GD Solution", fontsize=14)
        plt.legend(loc='best')
        plt.grid()
        plt.show()
    elif d>2:
        ("Plotting is not available for d>2, please check the results in the console.")
