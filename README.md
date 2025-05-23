# Difference of Max Affine (DoMA) Piecewise Linear Regression

This repository contains the implementation of the Difference of Max Affine (DoMA) Piecewise Linear Regression method, developed for the NeurIPS 2025 main Track. The DoMA method is designed to model complex, non-linear relationships within data by partitioning the input space into regions, each governed by its affine function. This piecewise approach allows for greater flexibility and precision in capturing underlying data patterns.

## Take a Trip Down Convergence Lane with ABGD
The solution we propose is ABGD (a variant of gradient descent) with a predetermined adaptive step size formula (***no hyperparameter tuning\!***). When the input dimension is `d`, only a sample size of `O(d)` is required for linear convergence (check Theorem 4.4). This result is established when the covariates, also called inputs, follow a subGaussian distribution (Gaussian, Uniform, bounded exponential, and many other multimodal densities), thus making it applicable to modern machine learning contexts that do not necessarily fit the typical Gaussian assumption. 

Next is an example of a piecewise linear function (blue), samples from this function (red), and the log l2 squared loss from ABGD showing linear decay.  
<p align="center">
  <img src="https://github.com/NeurIPS-2025-PL/DoMA-Piecewise-Linear-Regression/blob/a81d287f17a6a74295dd2c3a977ee7c032e0b775/Synthetic%20ABGD%20(Matlab)/DMax%20Figures/Linear_Con_Example.png" width="300" title="Linear convergence example">
</p>

## Repository contents:
The repository is structured into two main components:

  - **Synthetic ABGD**: This module focuses on generating synthetic datasets to validate the theoretical foundations of the ABGD method. It includes tools for data generation, model training, and evaluation.

  - **ABGD_Vs_The_World**: This module benchmarks the ABGD method acting on a DOMA function against existing piecewise linear regression techniques using real-world datasets (available from Matlab's Statistics & ML Toolbox). It provides comparative analyses to demonstrate the effectiveness and advantages of the DoMA+ABGD approach.

