# Numerical-Computation
The repository contains solution to steady-state heat conduction PDEs as well as 2D-transient heat conduction by the name of T2dHC solver (with multi-independent point BCs and domain inversion) .The equation are solved using explicit(2D-FDM explicit.py) and implicit(2D-FDM implicit solver.py) approach. T2dHC solver (with multi-independent point BCs and domain inversion) is solved using implicit approach .The PDEs are modelled using finite difference method and then solved further. The implict apprach solves the set of linear equation using Gauss-Seidel method to avoid the problem of sparse matrix.

T2dHC solver will require a little learning on how to operate,therefore I have put comments to guide user of what to change and what not to. But if then also anyone is not able to operate the solver so feel free to contact me at nabeelhasan661@gmail.com.

Gauss-Seidel solver:
Numpy and Sympy libraries do carry some solvers that are really good at solving system of linear equation. But they suffer with problems where system of linear equations becomes sparse; making the co-efficient of matrix singular i.e det(A) = 0. 
  Thus this solver here is capable of solving normal system of linear equations as well as system of linear equations equations that generate sparse matrices. This solver also allows the user to set the tolerance limit easily (currently set to 1.e-15).

9th March 2025 : I have also added a new solver for solving the incompressible fluid flows problem.
Now the repository heavely deals with numerical computation problems of heat-transfer and incompressible flows; using python based solvers. Below are the few results from these solvers.

More new updated and advanced version with user friendly system of editing BCs will be uploaded in near future.😀

Results from : T2dHC solver (with unit sized square changing domain)
https://github.com/user-attachments/assets/07b16b44-2d3a-4782-85a1-2b0dc5e3165f
Results from : Pressure correction poison equation solver model 3

![Figure_1](https://github.com/user-attachments/assets/7f078d5e-1e74-419e-9066-6baae3a7bed6)
Ghia et al data comparison (just for refrence)
![Screenshot 2025-04-03 005621](https://github.com/user-attachments/assets/89b862be-c90c-4753-bffb-9ddfd7a64ada)
