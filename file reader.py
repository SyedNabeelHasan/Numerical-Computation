import os 
from colorama import Fore, Style, init
import numpy as np
import pandas as pd


print("")
print(Fore.BLUE + "Welcome to .DAT file reader 📂" + Style.RESET_ALL)
print(f"NumPy version     : {np.__version__}")
print(f"Pandas version    : {pd.__version__}")
print("")

# choose start end point 
start_point = 1
end_point = 6

dir = []    # contains names of the files that are to be read
mtids = 3   # max_time_iteration_digit_size

folder_path = r"D:\POD_TRY"     # enter the folder path here
for i in range(start_point,end_point,1):
    filename = f"SNAP{i:0{mtids}d}.DAT"
    full_path = os.path.join(folder_path, filename)
    dir.append(full_path)


time = []
for do in dir:
    print(do)
    frame = []
    # Read file line by line and collect only numeric rows
    data = []
    with open(do, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("ZONE") or "I=" in line or "J=" in line or not line:
                continue
            try:
                values = list(map(float, line.split()))
                data.append(values)
            except ValueError:
                continue

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    # print(data[0])
    columns = list(zip(*data))

    for i in range(0,len(data[0]),1):
        frame.append(columns[i])
        # print(columns[i][1])

    time.append(frame)



print("just checking... ",time[0][1][3])
print("Numbers of frames extracted = ",len(time))


av_data = []  # the average data stores the required data

n_frames = len(time)
n_arrays = len(time[0])
n_values = len(time[0][0])

for b in range(n_arrays):  # loop over 8 arrays
    col = []
    for a in range(n_values):  # loop over elements inside each array
        sum_vals = []
        for c in range(n_frames):  # loop over frames
            sum_vals.append(time[c][b][a])
        avg = sum(sum_vals) / n_frames
        col.append(avg)
    av_data.append(col)
# print(av_data)


# file format builder algorithim 
avg_data_np = np.array(av_data).T
# Set desired I and J values (example: I = 217, J = 420)
I_val = 217
J_val = 420

# File save path
output_path = r"D:\POD_TRY\averaged_output.DAT"

with open(output_path, "w") as file:
    # Header
    file.write("ZONE\n")
    file.write(f"I= {I_val}\n")
    file.write(f"J= {J_val}\n")

    # Write data row by row
    for row in avg_data_np:
        line = "    ".join(f"{val:.8E}" for val in row)
        file.write(line + "\n")
print("")
print(Fore.YELLOW + "You averaged_output.DAT file is ready ✅" + Style.RESET_ALL)



