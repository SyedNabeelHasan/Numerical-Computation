import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
import time
from matplotlib.animation import FuncAnimation

st = time.time()

upperlimit = 1
lowerlimit = 0
total_time = 40
h = 0.05
del_t = 10

alpha = 0.0001

n = int((upperlimit-lowerlimit)/h)          
total_variables = (n-2)*(n-2)
print(n,"x",n,"and total number of variables = ",total_variables)

dc = time.time()
# mesh genration with no initial value
mesh = []
for i in range(0,n,1):
    f=[]
    for j in range(0,n,1):
        x = sp.symbols(f'x{i}|{j}')
        f.append(x)
    mesh.append(f)

# converting mesh into numpy operatable format
print(Fore.LIGHTMAGENTA_EX + "genrating mesh" + Style.RESET_ALL)
N_mesh = np.array(mesh)
print(N_mesh)
print("")
cd = time.time() ##############################

#set boundary conditions

N_mesh[0,:] = 27            # selects the entire first row of the mesh
N_mesh[n-1, :] = 27         # selects the entire  last row of the mesh
N_mesh[:, 0] = 27             # selects the entire first column of the mesh
N_mesh[:, n-1] = 27

P_mesh = np.zeros((n,n)) + 27   #time based initial boundary condition

P_mesh[0,:] = N_mesh[0,:]                
P_mesh[n-1,:] = N_mesh[n-1, :]           
P_mesh[:,0] = N_mesh[:, 0]              
P_mesh[:,n-1] = N_mesh[:, n-1]  
#print(P_mesh)

print(Fore.LIGHTMAGENTA_EX + "mesh after Boundary condition:" + Style.RESET_ALL)
#print(Xh)

print("")

r = (alpha*del_t)/(h**2)

qa = time.time()
#calculation using finite difference method
print(Fore.LIGHTMAGENTA_EX + "genrating equations: " + Style.RESET_ALL)

data_set = []

for no in np.arange(0,total_time,del_t):
    invert = 1      # set invert here 
    u = []
    index=[]
    inv_index = []  # BCs of inverted/-1 problem domain
    eqs = []
    
    func = int((no+del_t)/del_t) # Universal function (O/P natural nos)
    #define points(4-points given as of now)
    ay1 = 10
    bx1 = 10
    ay2 = 10
    bx2 = 11
    ay3 = 11
    bx3 = 10  
    ay4 = 11
    bx4 = 11

    location = [(ay1,bx1),(ay2,bx2),(ay3,bx3),(ay4,bx4)] # add location points here

    if (invert == 1):
        T_obj = 80
        N_mesh[ay1,bx1] = T_obj
        N_mesh[ay2,bx2] = T_obj 
        N_mesh[ay3,bx3] = T_obj
        N_mesh[ay4,bx4] = T_obj
        inv_BC_location = 0
    if (invert == -1 ):
        inv_BC_location = [(9,10),(9,11),(10,9),(10,12),(11,9),(11,12),(12,10),(12,11)]
        T_obj = 28
        N_mesh[9,10] = T_obj
        N_mesh[9,11] = T_obj 
        N_mesh[10,9] = T_obj
        N_mesh[10,12] = T_obj  
        N_mesh[11,9] = T_obj
        N_mesh[11,12] = T_obj 
        N_mesh[12,10] = T_obj
        N_mesh[12,11] = T_obj      
    
    for io in range(1, n-1,1):
        for jo in range(1, n-1,1):
            if ((io,jo) in location  and invert ==1):
                ind = ((io-1)*(n-2))+jo
                indi = ind - 1      # correction factor
                index.append(indi)
            elif (invert == 1):
                u.append(N_mesh[io][jo])
                eq = -1*r*(N_mesh[io+1][jo] + N_mesh[io][jo+1] + N_mesh[io-1][jo] + N_mesh[io][jo-1]) + (4*r + 1)*(N_mesh[io][jo]) - P_mesh[io][jo] 
                eqs.append(eq)
                sp.pretty_print(eq)
            if ((io,jo) in location  and invert == -1):
                u.append(N_mesh[io][jo])
                eq = -1*r*(N_mesh[io+1][jo] + N_mesh[io][jo+1] + N_mesh[io-1][jo] + N_mesh[io][jo-1]) + (4*r + 1)*(N_mesh[io][jo]) - P_mesh[io][jo] 
                eqs.append(eq)
                sp.pretty_print(eq) 
                ind = ((io-1)*(n))+jo
                indi = ind + 0          # correction factor
                index.append(indi)  
            elif  ( invert == -1 and (io,jo) in inv_BC_location ): 
                inde = ((io-1)*(n))+jo
                indee = inde + 0        # correction factor
                inv_index.append(indee)                 

    print("location of BC in inv -1 = ",inv_BC_location)
    print("their index = ",inv_index)
    
        #sp.pretty_print(eqs)
    aq = time.time() ################################
    
    
    nb = time.time()
    #genration of matrix A & B for solving sets of linear equation
    A_sub1 = []
    B_sub1 = []
    for a in range(0,len(eqs),1):
        expression = eqs[a]
        A_sub2 = []
        for b in range(0,len(u),1):
            e = u[b]
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
    print(A_np)
    print(B_np)

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
        tol = 1.e-15
        error = 2 * tol
        #print(s_inv)
        if (invert==1):
            x0 = np.zeros(((n-2)**2)-len(index))
        if (invert==-1):
            x0 = np.zeros(len(location))
            
        
        xsol = [x0]             # Initialization
        
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
    xv = time.time() ############################

    # Print the latest solution
    print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
    d = len(xsol)
    print(xsol[-1])
    array_u = np.array(xsol[-1]) # the solution exist in this array

    if (invert==1):
        array_u = np.array(xsol[-1])
        array_1 = np.insert(array_u,index[0],T_obj)
        array_2 = np.insert(array_1,index[1],T_obj)
        array_3 = np.insert(array_2,index[2],T_obj)
        array_4 = np.insert(array_3,index[3],T_obj)
        matrix2d =  array_4.reshape(n-2,n-2)
        for ui in range (1,n-1,1):
            for uj in range(1,n-1,1):
                P_mesh[ui][uj] = matrix2d[ui-1][uj-1]
        P_mesh[0,:] = N_mesh[0,:]                
        P_mesh[n-1,:] = N_mesh[n-1, :]           
        P_mesh[:,0] = N_mesh[:, 0]              
        P_mesh[:,n-1] = N_mesh[:, n-1] 

        print(index,no,a)
        N_mesh[ay1][bx1] = mesh[ay1][bx1]
        N_mesh[ay2][bx2] = mesh[ay2][bx2]
        N_mesh[ay3][bx3] = mesh[ay3][bx3]
        N_mesh[ay4][bx4] = mesh[ay4][bx4]
        data_set.append(np.array(P_mesh))
        print(index)
    
    if (invert == -1):
        array_li = P_mesh.ravel()
        for aa in range(0,len(inv_BC_location),1):
            ab = inv_index[aa]
            array_li[ab] = T_obj

        for ac in range(0,len(index),1):
            ad = index[ac]
            array_li[ad] = array_u[ac]

        P_inv_mesh = array_li.reshape(n,n)
        data_set.append(np.array(P_inv_mesh))

        
         

    print("-------P_mesh--------")
    print(P_mesh)
    
    
    print(Fore.BLUE + "mesh genration time = " + str(cd-dc) + Style.RESET_ALL)
    print(Fore.BLUE + "equation formation time = " + str(aq - qa) + Style.RESET_ALL)
    print(Fore.BLUE + "matrix formation time = " + str(bn-nb) + Style.RESET_ALL)
    print(Fore.BLUE + "SDD validation time = " + str(cf-fc) + Style.RESET_ALL)
    print(Fore.BLUE + "Gauss-Siedel evaluation time = " + str(xv-vx) + Style.RESET_ALL)

    print("--------new solution---------")




# # Create a figure and axis
fig, ax = plt.subplots()
x = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(n))
y = np.arange(lowerlimit, upperlimit, (upperlimit-lowerlimit)/(n))
X, Y = np.meshgrid(x, y)
# Initial Z and contour plot
Z = data_set[0]  # Example of initial Z
contour = ax.contourf(X, Y, Z, 20, cmap='viridis')
# Add a color bar once (outside the animation loop)
colorbar = plt.colorbar(contour, ax=ax, label='Temperature')

# Update function for each frame
def update(t):
    """Update the contour plot for frame t."""
    global contour  # Avoid creating new contour in each frame

    # Remove the previous contour plot by clearing the axis
    ax.clear()
 
    
    # New Z at each frame (you can change this to any time-dependent function)
    Z = data_set[t]
    
    # Create and update the contour plot
    contour = ax.contourf(X, Y, Z, 50, cmap='viridis')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', fontsize=12, fontweight='bold')
    time_text.set_text(f'Time: {(del_t * t)+del_t} s')
    return ax  # Return the new contour collections for updating

# Create the animation
ani = FuncAnimation(fig, update, frames=len(data_set), interval = 500, repeat=True)

# Display the plot
plt.show()


