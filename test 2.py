import numpy as np
import matplotlib.pyplot as plt
import time
from colorama import Fore, Style, init
from matplotlib.animation import FuncAnimation
import psutil

# Total and available memory (in GB)
total = psutil.virtual_memory().total / (1024**3)
available = psutil.virtual_memory().available / (1024**3)
print(Fore.GREEN + f"total space {total} GB" + Style.RESET_ALL)
print(Fore.RED + f"available space {available} GB" + Style.RESET_ALL)

st = time.time()

mesh_data = np.load(r"D:/numerical computation/geometry meshing/Meshes/RAX_1.npz")
ghost_nodes_data = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
inside_pt = mesh_data["array1"]
ghost_nodes = ghost_nodes_data["array2"]


del_h = 0.1
conversion_factor = 1/del_h     # mesh size
# conversion_factor = 
print("ooooo:::",inside_pt[0][0],inside_pt[0][1])
print(conversion_factor)


# nx = ny = int(10/del_h) + 1

nx = ny = 305
mat = np.full((nx, ny), np.nan, dtype=object)       # mesh for the solver

P_mat = np.full((nx, ny), np.nan)
P_mat_new = P_mat.copy()


variable_array = []
for i in range (0,len(inside_pt),1):

    ie,je = int(round((inside_pt[i][0]*conversion_factor),0)), int(round((inside_pt[i][1]*conversion_factor),0))
    x = f'x{je}|{ie}'
    variable_array.append(x)
print("🐒: ",len(variable_array))


print("Starting the solver...")

for i in range(0,len(inside_pt),1):
    x_coord=inside_pt[i][0]
    y_coord=inside_pt[i][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    mat[r][c] = variable_array[i]
    P_mat[r][c] = 27            # uniform initial boundary condition through out the geometry
    # print(r," " ,c)


# time.sleep(20)
Bc = [23, 580, 27, 270, 27,27,27,27,27,27,27,27]

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        if(ghost_nodes[i][j][0] == 0.98 and ghost_nodes[i][j][1] == 1.05):
            print(ghost_nodes[i][j][0],ghost_nodes[i][j][1],int(round((ghost_nodes[i][j][0]*conversion_factor),0)),(ghost_nodes[i][j][1]*conversion_factor))
            time.sleep(5)
        x = ghost_nodes[i][j][0]
        y = ghost_nodes[i][j][1]
        r = int(round((y * conversion_factor),0))
        c = int(round((x * conversion_factor),0))
        mat[r][c] = Bc[i]
        P_mat[r][c] = Bc[i]


# available = psutil.virtual_memory().available / (1024**3)
# print(Fore.RED + f"available space {available} GB" + Style.RESET_ALL)

print("ccc",mat[14][15])
# check the geometry 
plt.pcolormesh(P_mat,cmap = 'viridis')
plt.show()


r = (0.0001*10)/(0.01**2)
B_vector_sequence = []
data_set = []
for ret in np.arange(0,0.3,0.1):
    print(Fore.CYAN + "Time step = " + str(ret) + Style.RESET_ALL)
    #genration of matrix A & B for solving sets of linear equation
    ss = time.time()
    A = np.zeros((len(variable_array),len(variable_array)))
    B = []
    if(ret == 0):
        for i in range(0,len(variable_array),1):
            x_coord=inside_pt[i][0]
            y_coord=inside_pt[i][1]
            # At the point (x_coord, y_coord)
            row = int(round((y_coord * conversion_factor),0))
            col = int(round((x_coord * conversion_factor),0))  
            # Find the indices of the neighboring points
            east = col+1
            west = col-1
            south = row-1
            north = row+1  
            A[i][i] = 4*r + 1
            # Neighbor handling with safe check
            key_east = f'x{row}|{east}'
            key_west = f'x{row}|{west}'
            key_south = f'x{south}|{col}'
            key_north = f'x{north}|{col}'

            b_e = []
            bb_for_eqn = []
            if key_east in variable_array:
                east_m = variable_array.index(key_east)
                A[i][east_m] = -r
            else:
                b1 = mat[row][east] * (-1*r)
                b_e.append(b1)
                bb_for_eqn.append((row,east))

            if key_west in variable_array:
                west_m = variable_array.index(key_west)
                A[i][west_m] = -r
            else:
                b2 = mat[row][west] * (-1*r)
                b_e.append(b2)
                bb_for_eqn.append((row,west))

            if key_south in variable_array:
                south_m = variable_array.index(key_south)
                A[i][south_m] = -r
            else:
                b3 = mat[south][col] * (-1*r)
                b_e.append(b3)
                bb_for_eqn.append((south,col))

            if key_north in variable_array:
                north_m = variable_array.index(key_north)
                A[i][north_m] = -r
            else:
                b4 = mat[north][col] * (-1*r)
                b_e.append(b4)
                bb_for_eqn.append((north,col))
            if (len(bb_for_eqn) != 0):    
                B_vector_sequence.append(bb_for_eqn)
            else:
                B_vector_sequence.append(['un'])
                pass
            b_final = -1*((np.sum(b_e))-(P_mat[row][col]))
            B.append(b_final)
        B_np = np.array(B, dtype=float)
        A_np = np.array(A, dtype=float)
    print(B_vector_sequence[1])
    
    print("::::: ",len(B_vector_sequence))
    if(ret > 0):
        B = []
        for i in range(0,len(B_vector_sequence),1):
            row,col = int(inside_pt[i][1]*conversion_factor), int(inside_pt[i][0]*conversion_factor)
            b_e = []
            for j in range(0,len(B_vector_sequence[i]),1):
                #print(i,j)
                roww = B_vector_sequence[i][j][0]
                coll = B_vector_sequence[i][j][1]
                if (roww != 'u' and coll != 'n'):
                    # print("@@@@: ",roww,coll)
                    b_e.append((mat[roww][coll]) * (-1*r))
                if (roww == 'u' and coll == 'n'):
                    # print("$$$$:",roww,coll)
                    b_e.append(0)
            b_final = -1*((np.sum(b_e))-(P_mat[row][col]))
            B.append(b_final)
        B_np = np.array(B, dtype=float)

    print("🔥")
    print(B_np)
    print("🥶🥶🥶")
    print(A_np)
    print("=========================")
    # print(B_np)
    ee = time.time()
    print("Time taken for A & B matrix generation = ",ee-ss)
    #check for SDD
    # time.sleep(10)
    big_check = []
    for ic in range(0,len(variable_array),1):
        Akk = abs(A_np[ic][ic])
        sumofrow = []
        for jc in range(0,len(variable_array),1):
            Arest = abs(A_np[ic][jc])
            sumofrow.append(Arest)
        sum = np.sum(sumofrow) - abs(Akk)
        if (Akk>=sum):
            #print(Fore.GREEN + "SDD verified " + str(ic) + Style.RESET_ALL)
            big_check.append(1)
            
        if(Akk<sum):
            print(Fore.RED + "SDD failed " + str(ic) + Style.RESET_ALL)
            print(A_np[ic])
           

    if(len(big_check) == len(A_np)):
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

        x0 = np.zeros(len(variable_array))
        xsol = [x0]

        # Iterative loop
        for l in range(0, 1000000):
            
            xzee = s_inv@(B_np-(T@xsol[l])) 
            xsol.append(xzee)

            # Calculate the infinity norm of the difference between current and previous solution
            error = np.linalg.norm(xsol[l + 1] - xsol[l], ord=np.inf)
            #print("👉 ",l," ",error, xzee[1])

            # Check if error is below the tolerance
            if error < tol:
                print("Convergence achieved in iteration = ",l)
                break
    # Print the latest solution
    print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
    d = len(xsol)
    print(xsol[-1])

    solution_vector = np.array(xsol[-1], dtype=np.float64)
    for i in range(0,len(solution_vector),1):
        x_coord=inside_pt[i][0]
        y_coord=inside_pt[i][1]
        r = int(round((y_coord * conversion_factor),0))
        c = int(round((x_coord * conversion_factor),0))
        P_mat[r][c] = solution_vector[i]
        P_mat_new[r][c] = solution_vector[i]           # uniform initial boundary condition through out the geometry
        #print(P_mat)
    data_set.append(P_mat_new.copy())

et = time.time()
print("Total time taken ⏰ = ",et-st)
del_t = 0.1

# # Create a figure and axis
fig, ax = plt.subplots()
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
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
    contour = ax.contourf(X, Y, Z, 20, cmap='viridis')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='black', fontsize=12, fontweight='bold')
    time_text.set_text(f'Time: {(del_t * t)+del_t} s')
    return ax  # Return the new contour collections for updating

# Create the animation
ani = FuncAnimation(fig, update, frames=len(data_set), interval = 500, repeat=True)

# Display the plot
plt.show()


available = psutil.virtual_memory().available / (1024**3)
print(Fore.RED + f"available space {available} GB" + Style.RESET_ALL)

path1 = r"D:/numerical computation/geometry meshing/Meshes/Time_stack"
data_set = np.array(data_set, dtype=object)
np.savez(path1, array1=data_set)
