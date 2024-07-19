import numpy as np
import matplotlib.pyplot as plt
import time
from colorama import Fore, Style, init



start_time = time.time()
n = 50 # mesh size (don't exceed 1000)

# Initialize the mesh with 20 degrees
mesh = np.zeros((n, n)) + 20

print(Fore.LIGHTRED_EX + "Initial mesh:" + Style.RESET_ALL)
print(mesh)

# Define the x and y coordinates
x = np.arange(0, n, 1)
y = np.arange(0, n, 1)

# Set boundary values
mesh[0,:] = 700              # selects the entire first row of the mesh
#mesh[n-1, :] = 200             # selects the entire  last row of the mesh
#mesh[:, 0] = 700              # selects the entire first column of the mesh
#mesh[:, n-1] = 700          # selects the entire  last column of the mesh


print(Fore.LIGHTRED_EX + "Mesh with boundary values:" + Style.RESET_ALL )
print(mesh)

# Iterate to solve the Laplace equation
for _ in range(1000):  # Number of iterations
    u = mesh.copy()
    for i in range(1, n-1,1):
        for j in range(1, n-1,1):
            mesh[i][j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1])  # the laplace equation(considering dx = dy)
            #print("at",i,",",j)
            #print("0.25*(",u[i+1, j], "+", u[i-1, j], "+", u[i, j+1], "+", u[i, j-1],")")
    

print(Fore.GREEN + "Final mesh after iterations:" + Style.RESET_ALL)
print(mesh)
end_time = time.time()
duration = end_time - start_time
print(Fore.BLUE + "total time taken to evaluate the Laplace equation is",duration,"seconds" + Style.RESET_ALL)

# Plot the results
X, Y = np.meshgrid(x, y)
Z = mesh
plt.pcolormesh(X, Y, Z, shading='auto',cmap='viridis')
plt.colorbar(label='Temperature')
#plt.grid(True)
plt.title("Temperature Distribution")
plt.show()
