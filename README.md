# Numerical-Computation
The repository contains solution to steady-state heat conduction PDEs as well as 2D-transient heat conduction by the name of T2dHC solver (with multi-independent point BCs and domain inversion) .The equation are solved using explicit(2D-FDM explicit.py) and implicit(2D-FDM implicit solver.py) approach. T2dHC solver (with multi-independent point BCs and domain inversion) is solved using implicit approach .The PDEs are modelled using finite difference method and then solved further. The implict apprach solves the set of linear equation using Gauss-Seidel method to avoid the problem of sparse matrix.

T2dHC solver will require a little learning on how to operate,therefore I have put comments to guide user of what to change and what not to. But if then also anyone is not able to operate the solver so feel free to contact me at nabeelhasan661@gmail.com.

Gauss-Seidel solver:
Numpy and Sympy libraries do carry some solvers that are really good at solving system of linear equation. But they suffer with problems where system of linear equations becomes sparse; making the co-efficient of matrix singular i.e det(A) = 0. 
  Thus this solver here is capable of solving normal system of linear equations as well as system of linear equations equations that generate sparse matrices. This solver also allows the user to set the tolerance limit easily (currently set to 1.e-15).

More new updated and advanced version with user friendly system of editing BCs will be uploaded in near future.😀
