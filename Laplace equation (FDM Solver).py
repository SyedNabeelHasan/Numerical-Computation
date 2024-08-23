import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from colorama import Fore, Style, init


n = 12      #(for now dont exceed mesh size more than 12)

# mesh genration with no initial value
mesh = []
for i in range(0,n,1):
    f=[]
    for j in range(0,n,1):
        x = sp.symbols(f'x{j}{i}')
        f.append(x)
    mesh.append(f)

# converting mesh into numpy operatable format
print(Fore.LIGHTMAGENTA_EX + "mesh" + Style.RESET_ALL)
Xh = np.array(mesh)
print(Xh)
print("")

#set boundary conditions

Xh[0,:] = 70              # selects the entire first row of the mesh
Xh[n-1, :] = 70           # selects the entire  last row of the mesh
Xh[:, 0] = 20             # selects the entire first column of the mesh
Xh[:, n-1] = 20

print(Fore.LIGHTMAGENTA_EX + "mesh after Boundary condition:" + Style.RESET_ALL)
print(Xh)

#selecting the variables out
u = []
for i in range(1,n-1,1):
    for j in range(1,n-1,1):
        x = sp.symbols(f'x{i}{j}')
        u.append(x)
print(u)

print("")

#calculation using finite difference method
print(Fore.LIGHTMAGENTA_EX + "The equations: " + Style.RESET_ALL)
eqs = []
for io in range(1, n-1,1):
    for jo in range(1, n-1,1):
        eq = -4*(Xh[io,jo]) + Xh[io-1,jo] + Xh[io+1,jo] + Xh[io,jo-1] + Xh[io,jo+1] 
        eqs.append(eq)
        sp.pretty_print(eq)

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
print(Fore.LIGHTMAGENTA_EX + "matrix A:" + Style.RESET_ALL)
print(A_np)
print(Fore.LIGHTMAGENTA_EX + "matrix B:" + Style.RESET_ALL)
print(B_np)
#solving matrix A & B using gauss elimination method
solution = np.linalg.solve(A_np, B_np)
print("")
#printing out solutions 
# print(Fore.YELLOW + "solutions: " + Style.RESET_ALL)
# for l in range(0,len(solution),1):
#     print(f"x{l} = ", solution[l])

#changing fromat of 'solution' array from 1D to 2D array
arr_2d = solution.reshape(n-2, n-2)
print(solution,arr_2d)


# the arr_2d format is not in correct format of their alignment in accordence to mesh,therefore arr_2d is converted to correct form that is ds format by interchanging the the elements across diagonal of the matrix
ds = np.zeros((n-2,n-2)) + 0
for yu in range(0,n-2,1):
    for gu in range(0,n-2,1):
        ds[yu][gu] = arr_2d[gu][yu]  
#print("ds:",ds)        

#substituting the values calculated; back to orignal mesh or Xh
for j in range(0,n-2,1):
    for g in range(0,n-2,1):
        Xh[j+1][g+1] = ds[j,g]

#converting the mesh into suitable format for matplotlib to function 
zeta = np.zeros((n,n)) + 0 
for ir in range(0,n,1):
    for er in range(0,n,1):
        zeta[ir,er] = Xh[ir,er]

print(Fore.LIGHTMAGENTA_EX + "final mesh after evaluation:" + Style.RESET_ALL)
print(zeta)

#plotting out the graph/mesh
x = np.arange(0, n, 1)
y = np.arange(0, n, 1)
X, Y = np.meshgrid(x, y)
Z = zeta
plt.pcolormesh(X, Y, Z, shading='auto',cmap='viridis')
plt.colorbar(label='Temperature')
#plt.grid(True)
plt.title("Temperature Distribution")
plt.show()


