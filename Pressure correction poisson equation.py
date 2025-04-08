
# Note : On copy-pen the x axis (horizontal axis) is denoted with "i" and y axis (vertical axis) is denoted by "j" and the mode we generally write our cordinates is "(x,y)" format. So on page if we intend to travel right and left we will do i+1 and i-1 respectively and similarly for up and down travel we will do j+1 and j-1 respectively.

# But here in this solver the mode is row and cloumn that is (r,c); where "r" is vertical axis and denoted by "i" and "c" is horizontal axis denoted by "j"...so if u want to travel right of the grid it is j+1 and similarly j-1 for left. But if u intend to travel up and down in the grid it is i+1 and 1-1 respectively.



import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
import time
from matplotlib.animation import FuncAnimation

st = time.time()

upperlimit = 1      # define size of domain
lowerlimit = 0
total_time = 0.005  # set total simulation run time
h = 0.02           # set element size here
del_t = 0.001


n = int((upperlimit-lowerlimit)/h)          
total_variables = (n-2)*(n-2)
print(n,"x",n,"and total number of variables = ",total_variables)



# mesh generation for pressure calculation with no initial value
print(Fore.LIGHTMAGENTA_EX + "generating mesh_p..." + Style.RESET_ALL)

mesh_p = []
for i in range(0,n,1):
    f=[]
    for j in range(0,n,1):
        x = sp.symbols(f'x{i}|{j}')
        f.append(x)
    mesh_p.append(f)

# converting mesh into numpy operatable format
N_mesh_p = np.array(mesh_p)
print(Fore.LIGHTMAGENTA_EX + "mesh_p generation completed" + Style.RESET_ALL)
# print(N_mesh_p)
print("")

# initial mesh where pressure is zero everywehre at t = 0
p = np.zeros((n,n)) 

# velocity mesh at t = 0
u = np.zeros((n,n)) 
# print(u)
v = np.zeros((n,n)) 
# print("")


#------------------------------------------------------- set Boundary conditions-------------------------------#
ch_velocity = 100
kinematic_viscosity = 0.1
Re = 100

# u_non_dim = 10/10 = 1

#---set up velocity BC-----#
u[0,:] = 0
u[n-1,:] = 1  # visually on top  (lid top velocity)
u[:,0] = 0
u[:,n-1] = 0

v[0,:] = 0
v[n-1,:] = 0  # visually on top  
v[:,0] = 0
v[:,n-1] = 0

# print(u)
#---- set pressure BC ----#
N_mesh_p[0,:] = N_mesh_p[1,:]
N_mesh_p[n-1,:] = 0 # visually on top  
N_mesh_p[:,0] = N_mesh_p[:,1]
N_mesh_p[:,n-1] = N_mesh_p[:,n-2]

#----------------------------------------------------Main Algorithim-------------------------------------------#
u_star_stack = []
v_star_stack = []
u_stack = []
v_stack = []
p_stack = []

u_stack.append(u)
v_stack.append(v)
p_stack.append(p)

for ret in np.arange(0,total_time,del_t):
    t = int(ret/del_t)
    print("ret = ",ret," ","t = ",t)

    #-------------- calculating u* and v* -----------------#
    u_star_copy = u.copy()
    v_star_copy = v.copy()

    for ip in range(1,n-1,1):
        for jp in range(1,n-1,1):
            queta = u_stack[t][ip,jp] + del_t*((u_stack[t][ip,jp]*((u_stack[t][ip,jp+1] - u_stack[t][ip,jp-1])/(2*h))) + (v_stack[t][ip,jp]*((u_stack[t][ip+1,jp] - u_stack[t][ip-1,jp])/(2*h)))) - del_t*((p_stack[t][ip,jp+1] - p_stack[t][ip,jp-1])/(2*h)) + (del_t/Re)*((u_stack[t][ip,jp+1] + u_stack[t][ip,jp-1] + u_stack[t][ip+1,jp] + u_stack[t][ip-1,jp] - 4*u_stack[t][ip,jp])/(h*h))
            u_star_copy[ip,jp] = queta

            qveta = v_stack[t][ip,jp] + del_t*((u_stack[t][ip,jp]*((v_stack[t][ip,jp+1] - v_stack[t][ip,jp-1])/(2*h))) + (v_stack[t][ip,jp]*((v_stack[t][ip+1,jp] - v_stack[t][ip-1,jp])/(2*h)))) - del_t*((p_stack[t][ip+1,jp] - p_stack[t][ip-1,jp])/(2*h)) + (del_t/Re)*((v_stack[t][ip,jp+1] + v_stack[t][ip,jp-1] + v_stack[t][ip+1,jp] + v_stack[t][ip-1,jp] - 4*v_stack[t][ip,jp])/(h*h))
            v_star_copy[ip,jp] = qveta
    u_star_stack.append(u_star_copy)
    v_star_stack.append(v_star_copy)

    mesh_variables = []
    eqs = []
    for io in range(1,n-1,1):
        for jo in range(1,n-1,1):
            mesh_variables.append(N_mesh_p[io,jo])
            eq = N_mesh_p[io+1][jo] + N_mesh_p[io-1][jo] -4*(N_mesh_p[io,jo]) + N_mesh_p[io][jo+1] + N_mesh_p[io][jo-1] - ((h/(2*del_t))*((u_star_stack[t][io,jo+1]-u_star_stack[t][io,jo-1]) +  (v_star_stack[t][io+1,jo]-v_star_stack[t][io-1,jo])))
            eqs.append(eq)

    
    # matrix formation
    A_sub1 = []
    B_sub1 = []
    for a in range(0,len(eqs),1):
        expression = eqs[a]
        A_sub2 = []
        for b in range (0,len(mesh_variables),1):
            variable = mesh_variables[b]
            ef = expression.coeff(variable)
            A_sub2.append(ef)
        A_sub1.append(A_sub2)
        coeff_dict = expression.as_coefficients_dict()
        constant_term = coeff_dict[1]
        B_sub1.append(constant_term*(-1))


    A = np.array(A_sub1)
    B = np.array(B_sub1)
    print("")
    #print(A)
    #print(B)
    # converting matrix into numpy operatable form
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float)

    # SDD validation check
    big_check = []
    for ic in range(0,len(eqs),1):
        A_kk = abs(A_np[ic][ic])
        sumofrow = []
        for jc in range(0,len(eqs),1):
            A_rest = abs(A_np[ic][jc])
            sumofrow.append(A_rest)
        sum = np.sum(sumofrow) - abs(A_kk)
        if (A_kk>=sum):
            #print(Fore.GREEN + "SDD verified " + str(ic) + Style.RESET_ALL)
            big_check.append(1)
        
        if(A_kk<sum):
            print(Fore.RED + "SDD failed " + str(ic) + Style.RESET_ALL)
            print(A_np[ic])
            print(eqs[ic])
    
    # Gauss-Seidel solver
    if(len(big_check) == len(eqs)):
        print(Fore.GREEN + "-----Matrix A successfully follows SDD----- " + Style.RESET_ALL)
        # gauss siedel method
        s = np.tril(A_np)
        s_inv = np.linalg.inv(s)
        D = np.diag(np.diag(A_np))
        T = np.triu(A_np)-D
        print("IN THE LOOP")
        # Initial setup
        tol = 1.e-6
        error = 2 * tol
        #print(s_inv)

        x0 = np.zeros((n-2)**2)
        xsol = [x0]

        # Iterative loop
        for l in range(0, 100000000000000):
            
            xzee = s_inv@(B_np-(T@xsol[l])) 
            xsol.append(xzee)

            # Calculate the infinity norm of the difference between current and previous solution
            error = np.linalg.norm(xsol[l + 1] - xsol[l], ord=np.inf)
            print(f"L:{l} error = ",error)
            # Check if error is below the tolerance
            if error < tol:
                print("Convergence achieved in iteration = ",l)
                break

    # Print the latest solution
    print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
    d = len(xsol)
    # print(xsol[-1])
    # answer conversion in 2D format
    pressure_array_2d = xsol[-1]
    matrix2d = pressure_array_2d.reshape(n-2,n-2)
    p_npo = p.copy()
    for ie in range(1,n-1,1):
        for je in range(1,n-1,1):
            pressure = matrix2d[ie-1,je-1] 
            p_npo[ie,je] = pressure + p_stack[t][ie][je]        # p(n+1) = p' + p(n)
    
    p_npo[0,:] = p_npo[1,:]
    p_npo[n-1,:] = 0 # visually on top  
    p_npo[:,0] = p_npo[:,1]
    p_npo[:,n-1] = p_npo[:,n-2]

    p_stack.append(p_npo) # putting the final pressure values into p_stack 
    u_copy = u.copy()
    v_copy = v.copy()

    v_check=[]  #just for end velocity check
    for ib in range(1,n-1,1):
        for jb in range(1,n-1,1):
            queta_c = u_stack[t][ib,jb] + del_t*((u_stack[t][ib,jb]*((u_stack[t][ib,jb+1] - u_stack[t][ib,jb-1])/(2*h))) + (v_stack[t][i,j]*((u_stack[t][ib+1,jb] - u_stack[t][ib-1,jb])/(2*h)))) - del_t*((p_stack[t+1][ib,jb+1] - p_stack[t+1][ib,jb-1])/(2*h)) + (del_t/Re)*((u_stack[t][ib,jb+1] + u_stack[t][ib,jb-1] + u_stack[t][ib+1,jb] + u_stack[t][ib-1,jb] - 4*u_stack[t][ib,jb])/(h*h))
            u_copy[ib,jb] = queta_c
            # for Gha et al results change jb and t here...
            if (jb == 15 and t == 3 ):      
                v_check.append(queta_c)

            qveta_c = v_stack[t][ib,jb] + del_t*((u_stack[t][ib,jb]*((v_stack[t][ib,jb+1] - v_stack[t][ib,jb-1])/(2*h))) + (v_stack[t][ib,jb]*((v_stack[t][ib+1,jb] - v_stack[t][ib-1,jb])/(2*h)))) - del_t*((p_stack[t+1][ib+1,jb] - p_stack[t+1][ib-1,jb])/(2*h)) + (del_t/Re)*((v_stack[t][ib,jb+1] + v_stack[t][ib,jb-1] + v_stack[t][ib+1,jb] + v_stack[t][ib-1,jb] - 4*v_stack[t][ib,jb])/(h*h))
            v_copy[ib,jb] = qveta_c
    u_stack.append(u_copy)
    v_stack.append(v_copy)

    
et = time.time()

print("total time = ",et-st)
# sp.pretty_print(u_stack[1])
# print("")
# sp.pretty_print(v_stack[1])
# print("")
# sp.pretty_print(p_stack[1])

x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(n))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(n))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
Z = u_stack[-1]  # Example of initial Z
fig, ax = plt.subplots(figsize=(8,6))
contour = ax.contourf(X, Y, u_stack[-1], 20, cmap='coolwarm')
plt.colorbar(contour, ax=ax, label='u Velocity')
ax.streamplot(X, Y, u_stack[-1], v_stack[-1], color= 'k', density=1.5, linewidth=1)
plt.title("Lid Driven Cavity: Velocity Streamlines")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


y = np.linspace(lowerlimit, upperlimit, len(v_check))
v = v_check
plt.plot(v, y, marker='o', linestyle='-', color='b', markersize=5, label="Velocity Profile")
plt.xlabel("Velocity (v)")
plt.ylabel("Position (y)")
plt.show()