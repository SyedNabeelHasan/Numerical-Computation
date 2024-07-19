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












# cmap supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'