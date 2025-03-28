import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from colorama import Fore, Style, init

# Numpy and Sympy libraries do carry some solvers that are really good at solving system of linear equation. But they suffer with problems where system of linear equations becomes sparse; making the co-efficient of matrix singular i.e det(A) = 0. 
#   Thus this solver here is capable of solving normal system of linear equations as well as system of linear equations equations that generate sparse matrices. This solver also allows the user to set the tolerance limit easily (currently set to 1.e-15).

#------------------------------------------User Input Data-----------------------------------------------------#
A_sub1 = [[5,1,1,1,1],[1,5,1,1,1],[1,1,5,1,1],[1,1,1,5,1],[1,1,1,1,5]]

B_sub1 = [19,23,27,31,35]
#-------------------------------------------Gauss-Seidel Algorithim--------------------------------------------#

A = np.array(A_sub1)
B = np.array(B_sub1)
print("")
print("Matrix A")
print(A)
print("Matrix B")
print(B)
# converting matrix into numpy operatable form
A_np = np.array(A, dtype=float)
B_np = np.array(B, dtype=float)

# SDD validation check
big_check = []
for ic in range(0,len(A_sub1),1):
    A_kk = abs(A_np[ic][ic])
    sumofrow = []
    for jc in range(0,len(A_sub1),1):
        A_rest = abs(A_np[ic][jc])
        sumofrow.append(A_rest)
        sum = np.sum(sumofrow) - abs(A_kk)
    if (A_kk>=sum):
        #print(Fore.GREEN + "SDD verified " + str(ic) + Style.RESET_ALL)
        big_check.append(1)
        
    if(A_kk<sum):
        print(Fore.RED + "SDD failed " + str(ic) + Style.RESET_ALL)
        print(A_np[ic])
            
    
# Gauss-Seidel solver
if(len(big_check) == len(A_sub1)):
    print(Fore.GREEN + "-----Matrix A successfully follows SDD----- " + Style.RESET_ALL)
    # gauss siedel method
    s = np.tril(A_np)
    s_inv = np.linalg.inv(s)
    D = np.diag(np.diag(A_np))
    T = np.triu(A_np)-D
    
    # Initial setup
    tol = 1.e-15
    error = 2 * tol
    #print(s_inv)

    x0 = np.zeros(len(A_sub1))
    xsol = [x0]

    # Iterative loop
    for l in range(0, 100000000000000):
        xzee = s_inv@(B_np-(T@xsol[l])) 
        xsol.append(xzee)

        # Calculate the infinity norm of the difference between current and previous solution
        error = np.linalg.norm(xsol[l + 1] - xsol[l], ord=np.inf)

        # Check if error is below the tolerance
        if error < tol:
            print("Convergence achieved in iteration = ",l)
            break

    # Print the latest solution
print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
d = len(xsol)
print(xsol[-1])