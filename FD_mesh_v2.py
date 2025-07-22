import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import matplotlib.patches as patches



# Data storage
vertex = []
line = []  # Make sure this is defined globally
circle = []
boundary_coordinates_1 = []
horizontal = []
brk_id = []
# Setup figure
fig, ax = plt.subplots()
ax.set_title("Draw polygons (1), circles (2), arcs (3). Press 'q' to quit.")
plt.axis('equal')
plt.grid(True)

# State variables
current_mode = 'polygon'  # 'polygon', 'circle', or 'arc'
current_polygon = []
drawing_polygon = True

# Tkinter root (for dialogs)
root = tk.Tk()
root.withdraw()

def autoscale():
    ax.relim()
    ax.autoscale_view()
    ax.set_aspect('equal', adjustable='datalim')
    fig.canvas.draw()

def draw_polygon(points, style='b-'):
    xs, ys = zip(*points)
    ax.plot(xs, ys, style)
    autoscale()

def draw_circle(center, radius):
    circle = patches.Circle(center, radius, edgecolor='green', facecolor='none', linestyle='--')
    ax.add_patch(circle)
    autoscale()

def draw_arc(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    mid = (p1 + p2) / 2
    radius = np.linalg.norm(p1 - p2) / 2
    theta1 = np.degrees(np.arctan2(p1[1] - mid[1], p1[0] - mid[0]))
    theta2 = np.degrees(np.arctan2(p2[1] - mid[1], p2[0] - mid[0]))
    arc = patches.Arc(mid, 2 * radius, 2 * radius, angle=0, theta1=theta1, theta2=theta2, color='purple')
    ax.add_patch(arc)
    autoscale()

def onkey(event):
    global current_polygon, vertex, current_mode

    if event.key == 'q':
        if current_polygon:
            vertex.append(('polygon', current_polygon.copy()))
        print("Final data:\n", vertex)
        plt.close()

    elif event.key == 'c' and current_mode == 'polygon':
        if len(current_polygon) > 2:
            xs, ys = zip(*[current_polygon[-1], current_polygon[0]])
            ax.plot(xs, ys, 'b--')
            vertex.append(('polygon', current_polygon.copy()))
            current_polygon = []
            autoscale()

    elif event.key == 'escape':
        current_polygon = []
        fig.canvas.draw()

    elif event.key == '1':
        current_mode = 'polygon'
        print("Switched to polygon mode.")

    elif event.key == '2':
        current_mode = 'circle'
        try:
            input_str = simpledialog.askstring("Input Circle", "Enter center_x,center_y,radius:")
            cx, cy, r = map(float, input_str.strip().split(','))
            draw_circle((cx, cy), r)
            circle.append((cx, cy, r))
            vertex.append(('circle', cx, cy, r))
        except:
            print("Invalid input for circle.")

    elif event.key == '3':
        current_mode = 'arc'
        try:
            input_str = simpledialog.askstring("Input Arc", "Enter x1,y1 x2,y2:")
            parts = input_str.strip().split()
            p1 = tuple(map(float, parts[0].split(',')))
            p2 = tuple(map(float, parts[1].split(',')))
            draw_arc(p1, p2)
            vertex.append(('arc', p1, p2))
        except:
            print("Invalid input for arc.")

    elif event.key == 'i':
        try:
            input_str = simpledialog.askstring("Manual Input", "Enter points (e.g., 0,0 1,0 1,1 0,1 break):")
            parts = input_str.strip().split()

            pts = []
            

            for i, p in enumerate(parts):
                if p.lower() == "break":
                    pts.append("BREAK")
                    brk_id.append(len(pts) - 1)  # Position of "BREAK" in pts
                    break  # Optional: stop input here
                else:
                    x, y = map(float, p.strip().split(','))
                    pts.append((x, y))

            # Only draw valid points, ignore "BREAK"
            clean_pts = [p for p in pts if p != "BREAK"]
            draw_polygon(clean_pts, style='g-')

            line.extend(pts)  # Keep raw data, including "BREAK"
            vertex.append(('polygon', clean_pts))  # Store clean version
            print("Line data updated:", line)
            print("BREAK index positions:", brk_id)

        except Exception as e:
            print("Invalid input for manual polygon:", e)
# Connect events

fig.canvas.mpl_connect('key_press_event', onkey)

plt.show()
print(line)

# Calcuation of mid-boundary coordinates
del_h = 0.1
tol = 8
tolerance = 1e-8
space_laps = 1e-4

# calculation for line

for i in range(0, len(line)-1, 1):
    X1, X2 = line[i][0], line[i+1][0]
    Y1, Y2 = line[i][1], line[i+1][1]
    
    
    if ((X1 != "B" or Y1 !="R") and (X2 != "B" or Y2 !="R") ):
        
        print("👺: ",X1,Y1," ",X2,Y2)
        if (abs(X2 - X1) < 1e-5):  # consider vertical line
            # Use np.arange on Y, keep X constant        
            if (Y2 > Y1):
                for y in np.arange(Y1, Y2, space_laps):
                    boundary_coordinates_1.append((X1, y))
            elif (Y2 < Y1):
                for y in np.arange(Y1, Y2, -(space_laps)):
                    boundary_coordinates_1.append((X1, y))


        elif (abs(Y2-Y1) < 1e-5):     # consider horizontal line (simply these lines are not required in the algorithim to define if a point is in or out of domain.
            boundary_coordinates_1.append((X1,Y1))
            if(X2>X1):
                for x in np.arange(X1,X2,space_laps):
                    horizontal.append((x,Y1))
                

            elif(X1>X2):
                boundary_coordinates_1.append((X2,Y2))
                for x in np.arange(X1,X2,-(space_laps)):
                    horizontal.append((x,Y1))

        else:
            slope = (Y2 - Y1) / (X2 - X1)
            if (Y2 > Y1):
                print("")
                for y in np.arange(Y1 , Y2, space_laps ):
                    X = (((y - Y1)/slope) + X1)
                    boundary_coordinates_1.append((X, y))
                    #print("",(X,y))
        
            elif (Y1 > Y2):
                print("")
                for y in np.arange(Y1, Y2, -(space_laps)):
                    X = (((y - Y1)/slope) + X1)
                    boundary_coordinates_1.append((X, y))
                    #print("🕊️",(X,y))
                    
                                  
    else:
        pass

print(circle)          


    
# Round the result
rounded_points = [(round(float(x), tol), round(float(y), tol)) for x, y in boundary_coordinates_1]
rounded_horizontal = [(round(float(x), tol), round(float(y), tol)) for x, y in horizontal]

unique_rounded_points = list(set(rounded_points))

s,d = zip(*unique_rounded_points)
plt.scatter(s,d , s=2)
plt.show()
#-----vertex odd-even check-----#
for ip in range(0,len(brk_id),1):
    
    if (ip>0):
        end_point = np.sum(brk_id[:ip+1]) + ip
        start_point = np.sum([brk_id[:ip]]) + ip
    elif(ip==0):
        start_point = 0
        end_point = brk_id[ip]
    vertices = []
    for ihm in range(start_point,end_point,1):
        vertices.append(line[ihm])

    beta = 0 

    vertices.pop(len(vertices)-1)      #*******************************************************************************************
    print(vertices)

    even_vertex = []
    for i in range(0,len(vertices),1):
        if(i < len(vertices)-1):
            if (vertices[i-1][1] < vertices[i][1] and vertices[i+1][1] < vertices[i][1]):
                even_vertex.append(vertices[i])
            elif(vertices[i-1][1] > vertices[i][1] and vertices[i+1][1] > vertices[i][1]):
                even_vertex.append(vertices[i])
            elif((abs(vertices[i-1][0]-vertices[i][0]) < 1e-5) and (abs(vertices[i+1][1]-vertices[i][1]) < 1e-5)   or (abs(vertices[i+1][0]-vertices[i][0]) < 1e-5) and (abs(vertices[i-1][1]-vertices[i][1]) < 1e-5) ): # right angle pair
                if( (vertices[i-1][1] > vertices[i][1]) and (vertices[i][1] > vertices[i+2][1])):
                    even_vertex.append(vertices[i])
                elif((vertices[i-2][1] < vertices[i][1]) and (vertices[i+1][1] > vertices[i][1])):
                    even_vertex.append(vertices[i])
                else:
                    pass
            else:
                pass   
        
        if(i==len(vertices)-1): #****************************************************************************************************
            if (vertices[i-1][1] < vertices[i][1] and vertices[beta][1] < vertices[i][1]):
                even_vertex.append(vertices[i])
            elif(vertices[i-1][1] > vertices[i][1] and vertices[beta][1] > vertices[i][1]):
                even_vertex.append(vertices[i]) #*************************************************************************************
    
    for im in range(0,len(even_vertex),1):
        unique_rounded_points.append(even_vertex[im]) 
 
print("even: ",even_vertex)

print("----------------------------------------")
print("u: ",unique_rounded_points)
print("l: ",line)


#############################################################################################
Points = []
h = del_h 

for x in np.arange(0,10,h):
    for y in np.arange(0,10,h):
        Points.append((x,y))

rounded_points_2 = [(round(float(x), tol), round(float(y), tol)) for x, y in Points] # giving same round of to points as given to boundary points 
points = list(set(rounded_points_2) - (set(rounded_horizontal)|set(rounded_points))) # removing points that lies on boundary and thus are not required to be analyzed. 


#------------------------------------ checking of odd-even intersections with respect to x----------------------------------------#
interior_points_x=[]
for Y in np.arange(0,10,h):

    edge_points_x = []     # Stores data relating to boundary coordinates at particular Y value
    test_point_x = []      # Stores test points from complete space that have same particular Y value

    for i in range(0,len(unique_rounded_points),1):
        if (abs(unique_rounded_points[i][1] - Y) < tolerance):
            edge_points_x.append(unique_rounded_points[i])
        
                
    for j in range(0,len(points),1):
        if(abs(points[j][1] - Y) < tolerance):
            test_point_x.append(points[j])

    for k in range(0,len(test_point_x),1):
        counter_x=[]
        for w in range(0,len(edge_points_x),1):
            if(test_point_x[k][0] < edge_points_x[w][0]):
                counter_x.append("1")

        rem = len(counter_x)
        if (rem % 2 !=0):
            interior_points_x.append(test_point_x[k])
        else:
            pass
       
        
#     print("edges:",edge_points_x)
#     print("check points",test_point_x)
# print("----------")
# print("interior points",interior_points_x)

filtered_interior_x = [point for point in interior_points_x if point not in unique_rounded_points]


print(filtered_interior_x)

# Unzip coordinates for plotting
x, y = zip(*unique_rounded_points)
if (len(rounded_horizontal) !=0):
    g, h = zip(*rounded_horizontal)
c, d = zip(*points)
a, b = zip(*filtered_interior_x)
yl,u = zip(*even_vertex)


# Plotting
# plt.plot(x, y, 'r-', linewidth=2, label='Polygon Boundary')  # Red line
plt.scatter(x, y, color='red', s=20, label='Vertices')       # Red markers
if (len(rounded_horizontal) !=0):
    plt.scatter(g, h, color='red', s=20, label='Vertices')
plt.scatter(c, d, color ="blue", s= 5)
plt.scatter(a, b, color = "black", s = 5)       # x-marker
# plt.scatter(g, h, color = "black", s = 5)       # y-marker
#plt.fill(x, y, alpha=0.2, color='red')                       # Light red fill
plt.scatter(yl,u,color='yellow',s =6)

# Annotations and labels
plt.title("Polygon Visualization with Red Markers", fontsize=14)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.axis('equal')  # Fix aspect ratio

plt.show()