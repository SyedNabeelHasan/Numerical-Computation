import numpy as np
import sympy as sp
import time
from colorama import Fore, Style, init
import matplotlib.pyplot as plt

start_time = time.time()
#  -------------Boundary value problem----------------  #

lowerLimit = 0          #lower limit of domain
upperLimit = 1          #upper limit of domain
n = 500               #number of iterations (including zero)
h = float((upperLimit - lowerLimit)/(n-1))
print(Fore.MAGENTA+"h = ",h)

print(Fore.RED+"Check the nodes here:" + Style.RESET_ALL)
u = []
for i in range(0,n,1):
    y = sp.symbols(f'y{i}')
    u.append(y)
sp.pretty_print(u)
print("")

#boundary condition 
u[0] = 2
u[n-1] = 2
sp.pretty_print(u)

x = np.linspace(lowerLimit,upperLimit,n)


print(Fore.RED+"The equations genrated :" + Style.RESET_ALL)
eqs = []
for j in range(1,n-1,1):
    eq = ((1/h**2)*(u[j+1] - (2*u[j]) + u[j-1])) - x[j] - u[j]    # change the eqution here
    eqs.append(eq)
    sp.pretty_print(eq)
print("")

A=[]
B=[]
for si in range(0,len(eqs),1):
    expression = eqs[si]
    a = []
    for d in range(1,len(eqs)+1,1):
        e = (u[d])
        ef = expression.coeff(e)
        a.append(ef)
    coeff_dict = expression.as_coefficients_dict()
    constant_term = coeff_dict[1]
    B.append(constant_term*(-1))
    A.append(a) 
print("")
A_np = np.array(A, dtype=float)
B_np = np.array(B, dtype=float)
print(Fore.RED+"Matrix A :" + Style.RESET_ALL)
print(A_np)
print("")
print(Fore.RED+"Matrix B :" + Style.RESET_ALL)
print(B_np)
solution = np.linalg.solve(A_np, B_np)
# Print the solution
print(Fore.GREEN + "Solution:"+ Style.RESET_ALL)
print(Fore.YELLOW + "...")

for l in range(0,len(solution),1):
    print(f"y{l+1} = ", solution[l])
    u[l+1] = solution[l]
#print(x,u,len(x),len(u))

# --------------end-------------- #
# End time
end_time = time.time()
# Calculate the duration
duration = end_time - start_time
print(Fore.BLUE + f"Time taken to evaluate the differential equation is",duration, "seconds")

#print(plt.style.available)             #check styles available in graphs 
plt.style.use('seaborn-v0_8-dark-palette')
plt.plot(x,u)
plt.grid(True)
plt.show() 




#Example1 : eq = ((1/h**2)*(u[j+1] - 2*u[j] + u[j-1]))-u[j] - (x[j])*(x[j]-4) 
#Example2 : eq = ((1/h**2)*(u[j+1] - 2*u[j] + u[j-1]))-u[j] - (x[j])*(x[j]-4)
#Example3 : eq = 0.25*(u[j+1] + u[j-1])
