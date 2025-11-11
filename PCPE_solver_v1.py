import numpy as np
import matplotlib.pyplot as plt
import time
from colorama import Fore, Style, init
from matplotlib.animation import FuncAnimation
import psutil
from scipy.sparse.linalg import cg

# Total and available memory (in GB)
total = psutil.virtual_memory().total / (1024**3)
available = psutil.virtual_memory().available / (1024**3)
print(Fore.GREEN + f"total space {total} GB" + Style.RESET_ALL)
print(Fore.RED + f"available space {available} GB" + Style.RESET_ALL)

st = time.time()

mesh_data = np.load(r"D:/numerical computation/geometry meshing/Meshes/RAX_1.npz")
ghost_nodes_data = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
sorted_first_interface = np.load(r"D:/numerical computation/geometry meshing/Meshes/GAX_1.npz", allow_pickle=True)
inside_pt = mesh_data["array1"]
ghost_nodes = ghost_nodes_data["array1"]
sorted_first_interface = sorted_first_interface["array7"]
ghost_nodes_list = ghost_nodes_list = [list(map(tuple, block)) for block in ghost_nodes]


del_h = 0.2                # ⬅️ set mesh element size here
conversion_factor = 1/del_h     # mesh size
# conversion_factor = 
print("ooooo:::",inside_pt[0][0],inside_pt[0][1])
print(conversion_factor)


# nx = ny = int(10/del_h) + 1

nx = ny = 33
mat = np.full((nx, ny), np.nan, dtype=object)       # mesh for the solver

# numeric 2D mesh

u_mat = np.full((nx, ny), np.nan)   # u_velocity

v_mat = np.full((nx, ny), np.nan)   # v_velocity

p_mat = np.full((nx, ny), np.nan)   # pressure



variable_array = []                                 # variable marker mesh (in use for pressure)
for i in range (0,len(inside_pt),1):
    x_coord=inside_pt[i][0]
    y_coord=inside_pt[i][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    x = f'x{r}|{c}'       # beware of row and column to x-y coordinate system
    variable_array.append(x)
print("🐒: ",len(variable_array),variable_array[0])


variable_array_copy = variable_array.copy()         # all the pressure BC editing is done on it. Make ghost_node value == first_interface

# Appending all the initial conditions to respective mesh nodes u, v and p (for fluid nodes)
for i in range(0,len(inside_pt),1):
    x_coord=inside_pt[i][0]
    y_coord=inside_pt[i][1]
    # print(x_coord,"",y_coord)
    r = int(round((y_coord * conversion_factor),0))
    c = int(round((x_coord * conversion_factor),0))
    u_mat[r][c] = 0                 # uniform initial velocity (u) condition through out the geometry
    v_mat[r][c] = 0                 # uniform initial velocity (v) condition through out the geometry
    p_mat[r][c] = 0                 # uniform initial pressure condition
    
   


# Appending boundary condition on ghost-nodes on respective u, v and p mesh

drich_u = [0,0,1,0]             # x direction velocity drichilit boundary condition
drich_v = [0,0,0,0]             # y direction velocity drichilit boundary condition
drich_p = [0,0,0,0]             # pressure drichilit boundary condition

for i in range(0,len(ghost_nodes),1):
    for j in range(0,len(ghost_nodes[i]),1):
        x = ghost_nodes[i][j][0]
        y = ghost_nodes[i][j][1]
        r = int(round((y * conversion_factor),0))
        c = int(round((x * conversion_factor),0))
        u_mat[r][c] = drich_u[i]
        v_mat[r][c] = drich_v[i]
        p_mat[r][c] = 0

# drich_pressure = [0]        # pressure drichilit boundary condition only on top lid 
# for i in range(0,1,1):
#     for j in range(0,len(ghost_nodes[i]),1):
#         x = ghost_nodes[i][j][0]
#         y = ghost_nodes[i][j][1]
#         r = int(round((y * conversion_factor),0))
#         c = int(round((x * conversion_factor),0))
#         p_mat[r][c] = drich_pressure[i] 


# setting up Neumen boundary condition (append the value c△n and if Neumen BC not available the write "NCN"...(neumen-condition-not-available) ) 
Ne_BC = [0,0,"NCN",0]       
# for i in range(1,len(ghost_nodes),1):
#     for j in range(0,len(ghost_nodes[i]),1):
#         x_g = ghost_nodes[i][j][0]
#         y_g = ghost_nodes[i][j][1]
#         r_g = int(round((y * conversion_factor),0))
#         c_g = int(round((x * conversion_factor),0))
#         x_f = sorted_first_interface[i][j][0]
#         y_f = sorted_first_interface[i][j][1]
#         r_f = int(round((y * conversion_factor),0))
#         c_f = int(round((x * conversion_factor),0))
#         p_mat[r_g][c_g] = variable_array_copy[r_f][c_f]

print(">><<")
print(u_mat)
# time.sleep(500)

u_star_stack = []
v_star_stack = []
u_stack = []
v_stack = []
p_stack = []


u_stack.append(u_mat)
v_stack.append(v_mat)
p_stack.append(p_mat)

total_time = 2
del_t = 0.01
h = del_h/10
Re = 100
sst = time.time()
#--------------------------------------------------------------------------------------------------------#
B_vector_sequence = []
for ret in np.arange(1,total_time,del_t):
    t = int(((ret - 1) / del_t) + 1)
    print("itn = ",t)
    u_star_copy = u_mat.copy()           # u_velocity copy mesh
    v_star_copy = v_mat.copy()           # v_velocity copy mesh

    for i in range(0,len(inside_pt),1):
        x_coord=inside_pt[i][0]
        y_coord=inside_pt[i][1]
        # print(x_coord,"",y_coord)
        ip = int(round((y_coord * conversion_factor),0))
        jp = int(round((x_coord * conversion_factor),0))
        # #------------------------------------------Convective flux-----------------------------------------------------------------------------------------------------------------#
        # Hu_conv = u_stack[t-1][i][j]*((u_stack[t-1][i][j+1] - u_stack[t-1][i][j-1])/(2*del_h)) + v_stack[t-1][i][j]*((u_stack[t-1][i][j+1] - u_stack[t-1][i][j-1]))/(2*del_h)
        # Hv_conv = u_stack[t-1][i][j]*((v_stack[t-1][i+1][j] - v_stack[t-1][i-1][j])/(2*del_h)) + v_stack[t-1][i][j]*((v_stack[t-1][i+1][j] - v_stack[t-1][i-1][j]))/(2*del_h)
        # #------------------------------------------Diffusive flux-------------------------------------------------------------------------------------------------------------------#
        # Hu_diffusion = (u_stack[t-1][i][j+1] + u_stack[t-1][i-1][j-1] + u_stack[t-1][i+1][j] + u_stack[t-1][i-1][j] - 4*u_stack[t-1][i][j])/(del_h**2)  
        # Hv_diffusion = (v_stack[t-1][i][j+1] + v_stack[t-1][i-1][j-1] + v_stack[t-1][i+1][j] + v_stack[t-1][i-1][j] - 4*v_stack[t-1][i][j])/(del_h**2) 
        # #-----------------------------------------Final flux (Up)-------------------------------------------------------------------------------------------------------------------#
        # Up = Hu_conv + Hu_diffusion 
        # Vp = Hv_conv + Hv_diffusion

        queta = u_stack[t-1][ip][jp] + del_t*((u_stack[t-1][ip][jp]*((u_stack[t-1][ip][jp+1] - u_stack[t-1][ip][jp-1])/(2*h))) + (v_stack[t-1][ip][jp]*((u_stack[t-1][ip+1][jp] - u_stack[t-1][ip-1][jp])/(2*h)))) - del_t*((p_stack[t-1][ip][jp+1] - p_stack[t-1][ip][jp-1])/(2*h)) + (del_t/Re)*((u_stack[t-1][ip][jp+1] + u_stack[t-1][ip][jp-1] + u_stack[t-1][ip+1][jp] + u_stack[t-1][ip-1][jp] - 4*u_stack[t-1][ip][jp])/(h*h))
        u_star_copy[ip][jp] = queta
        
        qveta = v_stack[t-1][ip][jp] + del_t*((u_stack[t-1][ip][jp]*((v_stack[t-1][ip][jp+1] - v_stack[t-1][ip][jp-1])/(2*h))) + (v_stack[t-1][ip][jp]*((v_stack[t-1][ip+1][jp] - v_stack[t-1][ip-1][jp])/(2*h)))) - del_t*((p_stack[t-1][ip+1][jp] - p_stack[t-1][ip-1][jp])/(2*h)) + (del_t/Re)*((v_stack[t-1][ip][jp+1] + v_stack[t-1][ip][jp-1] + v_stack[t-1][ip+1][jp] + v_stack[t-1][ip-1][jp] - 4*v_stack[t-1][ip][jp])/(h*h))
        v_star_copy[ip][jp] = qveta
        
    
    # for moving and deforming bodies
    # inside_pt = time_based_inside_pt[t]     # here inside points which are going to be function of time are stored in time_based_inside_pt
    # ghost_node = time_based_ghost_node[t]   # here ghost points which are going to be function of time are stored in time_based_ghost_node
    if(t == 1):
        sat = time.time()
        A = np.zeros((len(variable_array),len(variable_array)))
        B = []
        for i in range(0,len(variable_array),1):
            x_coord=inside_pt[i][0]
            y_coord=inside_pt[i][1]
            # At the point (x_coord, y_coord)
            # print("<M>",x_coord,y_coord)
            row = int(round((y_coord * conversion_factor),0))
            col = int(round((x_coord * conversion_factor),0)) 
            # print(">M<",row,col)
            io=row
            jo=col 
            # Find the indices of the neighboring points
            east = col+1
            west = col-1
            south = row-1
            north = row+1  
                    
            # Neighbor handling with safe check
            key_east = f'x{row}|{east}'
            key_west = f'x{row}|{west}'
            key_south = f'x{south}|{col}'
            key_north = f'x{north}|{col}'
            # print(key_east)
            # print(key_north)
            # print(key_south)
            # print(key_west)

            a = [-4]
            b_e = []
            b_vector_data = []
            if key_east in variable_array:
                east_m = variable_array.index(key_east)
                A[i][east_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t =  east/conversion_factor
                    y_t = row/conversion_factor
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC[ne_pos] != "NCN"):
                            b_e.append(Ne_BC[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC[ne_pos])
                            a.append(1)
                            if (x_coord == 2 and y_coord==2):
                                print("HI-1!!!!",Ne_BC[ne_pos])
                        else:
                            b_e.append(p_mat[row][east])
                            b_vector_data.append(p_mat[row][east])
                            pass
                    else:
                        pass
               
            if key_west in variable_array:
                west_m = variable_array.index(key_west)
                A[i][west_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = west / conversion_factor
                    y_t = row/ conversion_factor
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC[ne_pos] != "NCN"):
                            b_e.append(Ne_BC[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC[ne_pos])
                            a.append(1)
                            if (x_coord == 2 and y_coord== 2):
                                print("HI-2!!!!",Ne_BC[ne_pos])
                        else:
                            b_e.append(p_mat[row][west])
                            b_vector_data.append(p_mat[row][west])
                            pass
                    else:
                        pass
                      
            if key_south in variable_array:
                south_m = variable_array.index(key_south)
                A[i][south_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = col / conversion_factor
                    y_t = south/ conversion_factor
                    target = (x_t,y_t)
                    # need to convert target back into x-y coordinate
                    # print("down",target,key_south,key_west)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        # print(";;;;")
                        ne_pos = ghost_nodes_list.index(current_sub_gn)      # this line tells which edge we are dealng with
                        if (Ne_BC[ne_pos] != "NCN"):
                            b_e.append(Ne_BC[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC[ne_pos])
                            a.append(1)
                            if (x_coord == 2 and y_coord==2):
                                print("HI-3!!!!",Ne_BC[ne_pos])
                                print("NNNN: ",ne_pos)
                        else:
                            b_e.append(p_mat[south][col])
                            b_vector_data.append(p_mat[south][col])
                            pass
                    else:
                        pass
        
            if key_north in variable_array:
                north_m = variable_array.index(key_north)
                A[i][north_m] = 1
                b_vector_data.append(0)
            else:
                for ijx in range(0,len(ghost_nodes),1):
                    x_t = col / conversion_factor
                    y_t = north/ conversion_factor
                    target = (x_t,y_t)
                    current_sub_gn = ghost_nodes_list[ijx]
                    if target in current_sub_gn:
                        ne_pos = ghost_nodes_list.index(current_sub_gn)
                        if (Ne_BC[ne_pos] != "NCN"):
                            b_e.append(Ne_BC[ne_pos])       # appending c△n in B vector
                            b_vector_data.append(Ne_BC[ne_pos])
                            if (x_coord == 2 and y_coord== 2):
                                print("HI-4!!!!",Ne_BC[ne_pos])
                            a.append(1)                     # appending 1 in "a"
                        else:
                            b_e.append(p_mat[north][col])
                            b_vector_data.append(p_mat[north][col])
                            pass
                    else:
                        pass
      
            diag_a = np.sum(a)
            A[i][i] = diag_a
            # const = -1*((del_h)/(2*del_t)) * (u_star_copy[io+1][jo] - u_star_copy[io-1][jo] + v_star_copy[io][jo+1] + v_star_copy[io][jo-1])
            const = ((h/(2*del_t))*((u_star_copy[io][jo+1]-u_star_copy[io][jo-1]) +  (v_star_copy[io+1][jo]-v_star_copy[io-1][jo])))
            b_e.append(const)
            # if(x_coord== 2 and y_coord== 2):
            #     print(const,"//",io,jo,"////",b_e,"..<><><>",a)
            b_final = np.sum(b_e)
            B.append(b_final)
            B_vector_sequence.append(b_vector_data)
            # time.sleep(300)

        B_np = np.array(B, dtype=np.float64)
        A_np = np.array(A, dtype=np.float64)
        eat = time.time()
        print("A_time: ",eat-sat)
        s = np.tril(A_np)               # lower triangular matrix (contains the diagonal element)
        s_inv = np.linalg.inv(s)
        D = np.diag(np.diag(A_np))
        T = np.triu(A_np)-D             # STRICTLY upper triangular matrix (contains only element above the diagonal)

        
    # ul = len(B_np)
    # print("--------pop-------")
    # print("A matrix")
    # # print(A_np)
    # print("B vector")
    # # print(B_np)
    # print("variable array")
    # print(variable_array)
    
    
    elif(t > 1):
        sbt = time.time()
        B_sub = []
        for i in range(0,len(B_vector_sequence),1):
            x_coord=inside_pt[i][0]
            y_coord=inside_pt[i][1]
            row = int(round((y_coord * conversion_factor),0))
            col = int(round((x_coord * conversion_factor),0)) 
            # print(">M<",row,col)
            io=row
            jo=col 
            const = ((h/(2*del_t))*((u_star_copy[io][jo+1]-u_star_copy[io][jo-1]) +  (v_star_copy[io+1][jo]-v_star_copy[io-1][jo])))
            b = B_vector_sequence[i]
            b_sum = const + np.sum(b)
            B_sub.append(b_sum)
        B_np = np.array(B_sub, dtype=np.float64)
        ebt = time.time()
        print("B_time::",ebt-sbt)

    st = time.time()
    # solution_vector, info = cg(A_np, B_np, tol=1e-10, maxiter=5000)   # Conjugate gradient 
    #  print("Time taken for A & B matrix generation = ",ee-ss)
    # check for SDD
    # time.sleep(10)
    # big_check = []
    # for ic in range(0,len(variable_array),1):
    #     Akk = abs(A_np[ic][ic])
    #     sumofrow = []
    #     for jc in range(0,len(variable_array),1):
    #         Arest = abs(A_np[ic][jc])
    #         sumofrow.append(Arest)
    #     sum = np.sum(sumofrow) - abs(Akk)
    #     if (Akk>=sum):
    #         #print(Fore.GREEN + "SDD verified " + str(ic) + Style.RESET_ALL)
    #         big_check.append(1)
            
    #     if(Akk<sum):
    #         print(Fore.RED + "SDD failed " + str(ic) + Style.RESET_ALL)
    #         print(A_np[ic])
           

    # if(len(big_check) == len(A_np) ):
    #     print(Fore.GREEN + "-----Matrix A successfully follows SDD----- " + Style.RESET_ALL)
    #     # gauss siedel method
    #     # s = np.tril(A_np)               # lower triangular matrix (contains the diagonal element)
    #     # s_inv = np.linalg.inv(s)
    #     # D = np.diag(np.diag(A_np))
    #     # T = np.triu(A_np)-D             # STRICTLY upper triangular matrix (contains only element above the diagonal)

    #     tol = 1.e-6
    #     error = 2 * tol
    #     #print(s_inv)

    #     x0 = np.zeros(len(variable_array))
    #     xsol = [x0]

    #     # Iterative loop
    #     for l in range(0, 5000):
            
    #         xzee = s_inv@(B_np-(T@xsol[l]))     # the gauss-seidel general iterative formula
    #         xsol.append(xzee)

    #         # Calculate the infinity norm of the difference between current and previous solution
    #         error = np.linalg.norm(xsol[l + 1] - xsol[l], ord=np.inf)
    #         # print("👉 ",l," ",error)

    #         # Check if error is below the tolerance
    #         if error < tol:
    #             print("Convergence achieved in iteration = ",l)
    #             break
    #     solution_vector = np.array(xsol[-1], dtype=np.float64)
        # print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
        # print(solution_vector)
    
    solution_vector, info = cg(A_np, B_np, rtol = 1e-6, maxiter = 5000)   # Conjugate gradient
    print(Fore.YELLOW + "Final-Solution" + Style.RESET_ALL)
    print(solution_vector)
        
    et = time.time()
    print("Gauss-seidel time 🕛: ",et - st)
    

    p_mat_copy = p_mat.copy()                   # pressure copy mesh     # p(n+1) = p' + p(n)
    for i in range(0,len(solution_vector),1):
        x_coord=inside_pt[i][0]
        y_coord=inside_pt[i][1]
        r = int(round((y_coord * conversion_factor),0))
        c = int(round((x_coord * conversion_factor),0))
        #p_mat_copy[r][c] = solution_vector[i]
        p_mat_copy[r][c] = solution_vector[i] + p_stack[t-1][r][c]          # uniform initial boundary condition through out the geometry
       

    drich_p = [0,0,0,0]             # pressure drichilit boundary condition
    for i in range(0,len(ghost_nodes),1):
        for j in range(0,len(ghost_nodes[i]),1):
            x = ghost_nodes[i][j][0]
            y = ghost_nodes[i][j][1]
            r = int(round((y * conversion_factor),0))
            c = int(round((x * conversion_factor),0))
            p_mat_copy[r][c] = drich_p[i]


    p_mat_copy[0,:] = p_mat_copy[1,:]
    p_mat_copy[nx-1,:] = 0
    p_mat_copy[:,0] = p_mat_copy[:,1]
    p_mat_copy[:,nx-1] = p_mat_copy[:,nx-2]
    # print(p_mat_copy)
    p_stack.append(p_mat_copy) # putting the final pressure values into p_stack 

    u_copy = u_mat.copy()
    v_copy = v_mat.copy()
        
    # drich_u = [0,0,1,0]             # x direction velocity drichilit boundary condition
    # drich_v = [0,0,0,0]             # y direction velocity drichilit boundary condition

    # for i in range(0,len(ghost_nodes),1):
    #     for j in range(0,len(ghost_nodes[i]),1):
    #         x = ghost_nodes[i][j][0]
    #         y = ghost_nodes[i][j][1]
    #         r = int(round((y * conversion_factor),0))
    #         c = int(round((x * conversion_factor),0))
    #         u_copy[r][c] = drich_u[i]
    #         v_copy[r][c] = drich_v[i]
    #         # p_mat_copy[r][c] = drich_p[i]


    
    for i in range(0,len(inside_pt),1):
        x_coord=inside_pt[i][0]
        y_coord=inside_pt[i][1]
        # print(x_coord,"",y_coord)
        ib = int(round((y_coord * conversion_factor),0))
        jb = int(round((x_coord * conversion_factor),0))

        queta_c = u_stack[t-1][ib][jb] + del_t*((u_stack[t-1][ib][jb]*((u_stack[t-1][ib][jb+1] - u_stack[t-1][ib][jb-1])/(2*h))) + (v_stack[t-1][ib][jb]*((u_stack[t-1][ib+1][jb] - u_stack[t-1][ib-1][jb])/(2*h)))) - del_t*((p_stack[t][ib][jb+1] - p_stack[t][ib][jb-1])/(2*h)) + (del_t/Re)*((u_stack[t-1][ib][jb+1] + u_stack[t-1][ib][jb-1] + u_stack[t-1][ib+1][jb] + u_stack[t-1][ib-1][jb] - 4*u_stack[t-1][ib][jb])/(h*h))
        u_copy[ib,jb] = queta_c
        # for Gha et al results change jb and t here...

        qveta_c = v_stack[t-1][ib][jb] + del_t*((u_stack[t-1][ib][jb]*((v_stack[t-1][ib][jb+1] - v_stack[t-1][ib][jb-1])/(2*h))) + (v_stack[t-1][ib][jb]*((v_stack[t-1][ib+1][jb] - v_stack[t-1][ib-1][jb])/(2*h)))) - del_t*((p_stack[t][ib+1][jb] - p_stack[t][ib-1][jb])/(2*h)) + (del_t/Re)*((v_stack[t-1][ib][jb+1] + v_stack[t-1][ib][jb-1] + v_stack[t-1][ib+1][jb] + v_stack[t-1][ib-1][jb] - 4*v_stack[t-1][ib][jb])/(h*h))
        v_copy[ib,jb] = qveta_c

        # print("?/>",queta_c," ",qveta_c)
    
    
    u_stack.append(u_copy)
    v_stack.append(v_copy)
    # time.sleep(5000)
eet = time.time()
print("Net calculation time = ",eet-sst)
lowerlimit = 0
upperlimit = 1
n = nx

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


# y = np.linspace(lowerlimit, upperlimit, len(v_check))
# viz = v_check

# print("---------------")
# print(v_check)
# print("---------------")
# print(y)

# plt.plot(viz, y, marker='o', linestyle='-', color='b', markersize=5, label="Velocity Profile")
# plt.grid(True)
# plt.xlabel("Velocity (v)")
# plt.ylabel("Position (y)")
# plt.tight_layout()
# plt.show()

    


        
