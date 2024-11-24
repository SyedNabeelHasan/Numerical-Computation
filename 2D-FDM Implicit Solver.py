import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
import time


st = time.time()

upperlimit = 1
lowerlimit = 0
h = 0.02


n = int((upperlimit-lowerlimit)/h)      #(for now dont exceed mesh size more than 12)
print(n)

dc = time.time()
# mesh genration with no initial value
mesh = []
for i in range(0,n,1):
    f=[]
    for j in range(0,n,1):
        x = sp.symbols(f'x{j}|{i}')
        f.append(x)
    mesh.append(f)

# converting mesh into numpy operatable format
print(Fore.LIGHTMAGENTA_EX + "genrating mesh" + Style.RESET_ALL)
Xh = np.array(mesh)
#print(Xh)
print("")
cd = time.time() ##############################

#set boundary conditions

Xh[0,:] = 100              # selects the entire first row of the mesh
Xh[n-1, :] = 200           # selects the entire  last row of the mesh
Xh[:, 0] = 29            # selects the entire first column of the mesh
Xh[:, n-1] = 17

print(Fore.LIGHTMAGENTA_EX + "mesh after Boundary condition:" + Style.RESET_ALL)
#print(Xh)

#selecting the variables out
u = []
for j in range(1,n-1,1):
    for i in range(1,n-1,1):
        x = sp.symbols(f'x{i}|{j}')
        u.append(x)
#print(u)

print("")

qa = time.time()
#calculation using finite difference method
print(Fore.LIGHTMAGENTA_EX + "genrating equations: " + Style.RESET_ALL)
eqs = []
for io in range(1, n-1,1):
    for jo in range(1, n-1,1):
        eq = -4*(Xh[io,jo]) + Xh[io-1,jo] + Xh[io+1,jo] + Xh[io,jo-1] + Xh[io,jo+1] 
        eqs.append(eq)
        #sp.pretty_print(eq)
aq = time.time() ################################

nb = time.time()
#genration of matrix A & B for solving sets of linear equation
A_sub1 = []
B_sub1 = []
for a in range(0,len(eqs),1):
    expression = eqs[a]
    A_sub2 = []
    for b in range(0,len(u),1):
        e = (u[b])
        ef = expression.coeff(e) 
        #print(e,ef)
        A_sub2.append(ef)
    A_sub1.append(A_sub2)
    coeff_dict = expression.as_coefficients_dict()
    constant_term = coeff_dict[1]
    B_sub1.append(constant_term*(-1))
A = np.array(A_sub1)
B = np.array(B_sub1)
print("")
#converting matrix into numpy operatable form
A_np = np.array(A, dtype=float)
B_np = np.array(B, dtype=float)
bn = time.time() #################################

#print(A_np)
fc = time.time()
#check for SDD
big_check = []
for ic in range(0,len(eqs),1):
    Akk = abs(A_np[ic][ic])
    sumofrow = []
    for jc in range(0,len(eqs),1):
        Arest = abs(A_np[ic][jc])
        sumofrow.append(Arest)
    sum = np.sum(sumofrow) - abs(Akk)
    if (Akk>=sum):
        #print(Fore.GREEN + "SDD verified " + str(ic) + Style.RESET_ALL)
        big_check.append(1)
        
    if(Akk<sum):
        print(Fore.RED + "SDD failed " + str(ic) + Style.RESET_ALL)
        print(A_np[ic])
        print(eqs[ic])

cf = time.time() #############################


vx = time.time()
if(len(big_check) == len(eqs)):
    print(Fore.GREEN + "-----Matrix A successfully follows SDD----- " + Style.RESET_ALL)
    # gauss siedel method
    s = np.tril(A_np)
    s_inv = np.linalg.inv(s)
    D = np.diag(np.diag(A_np))
    T = np.triu(A_np)-D
    
    # Initial setup
    tol = 1.e-6
    error = 2 * tol
    #print(s_inv)

    x0 = np.zeros((n-2)**2)
    xsol = [x0]

    # Iterative loop
    for l in range(0, 220):
        xzee = s_inv@(B_np-(T@xsol[l])) 
        xsol.append(xzee)

        # Calculate the infinity norm of the difference between current and previous solution
        error = np.linalg.norm(xsol[l + 1] - xsol[l], ord=np.inf)

        # Check if error is below the tolerance
        if error < tol:
            print("Convergence achieved in iteration = ",l)
            break
xv = time.time() ############################

# Print the latest solution
print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
print(xsol[-1])

et = time.time()
print(Fore.BLUE + "mesh genration time = " + str(cd-dc) + Style.RESET_ALL)
print(Fore.BLUE + "equation formation time = " + str(aq - qa) + Style.RESET_ALL)
print(Fore.BLUE + "matrix formation time = " + str(bn-nb) + Style.RESET_ALL)
print(Fore.BLUE + "SDD validation time = " + str(cf-fc) + Style.RESET_ALL)
print(Fore.BLUE + "Gauss-Siedel evaluation time = " + str(xv-vx) + Style.RESET_ALL)
print(Fore.BLUE + "total time for evaluation = " + str(et-st) + Style.RESET_ALL)
