import matplotlib.pyplot as plt
import numpy as np

polygon = [
    (0,0), (0,0.5), (0,1), (0,1.5), (0,2), (0,2.5), (0,3),
    (0.5,3), (1,3), (1.5,3), (2,3), (2,2.5), (2,2.0), (2,1.5), (2,1),
    (2.5,1), (3,1), (3.5,1), (4,1), (4,1.5), (4,2), (4,2.5), (4,3),
    (4.5,3), (5,3), (5.5,3), (6,3), (6,2.5), (6,2.0), (6,1.5), (6,1),
    (6.5,1), (7,1), (7.5,1), (8,1), (8,1.5), (8,2.0), (8,2.5), (8,3),
    (8.5,3), (9,3), (9.5,3), (10,3), (10,2.5), (10,2), (10,1.5), (10,1),
    (10,0.5), (10,0), (9.5,0), (9,0), (8.5,0), (8,0), (7.5,0), (7,0),
    (6.5,0), (6,0), (5.5,0), (5,0), (4.5,0), (4,0), (3.5,0), (3,0),
    (2.5,0), (2.0,0), (1.5,0), (1.0,0), (0.5,0)
]


print(polygon[1][1])

points = []
h = 0.5 

for x in np.arange(0,10,h):
    for y in np.arange(0,10,h):
        points.append((x,y))

#------------------------------------ checking of odd-even intersections with respect to x----------------------------------------#
interior_points_x=[]
for Y in np.arange(0,3,h):

    edge_points_x = []     # Stores data relating to boundary coordinates at particular Y value
    test_point_x = []      # Stores test points from complete space that have same particular Y value

    for i in range(0,len(polygon),1):
        if(polygon[i][1] == Y):
            edge_points_x.append(polygon[i])
    
    for j in range(0,len(points),1):
        if(points[j][1] == Y):
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
       
        
    print("edges:",edge_points_x)
    print("check points",test_point_x)
print("----------")
print("interior points",interior_points_x)

filtered_interior_x = [point for point in interior_points_x if point not in polygon]
print(filtered_interior_x)

#--------------------------- checking of odd-even intersections with respect to y---------------------------------------------------#
interior_points_y=[]
for X in np.arange(0,10,h):

    edge_points_y = []      # Stores data relating to boundary coordinates at particular X value
    test_point_y = []       # Stores test points from complete space that have same particular X value

    for io in range(0,len(polygon),1):
        if(polygon[io][0] == X):
            edge_points_y.append(polygon[io])
    
    for jo in range(0,len(points),1):
        if(points[jo][0] == X):
            test_point_y.append(points[jo])

    for ko in range(0,len(test_point_y),1):
        counter_y=[]
        for wo in range(0,len(edge_points_y),1):
            if(test_point_y[ko][1] < edge_points_y[wo][1]):
                counter_y.append("1")

        rem = len(counter_y)
        if (rem % 2 !=0):
            interior_points_y.append(test_point_y[ko])
        else:
            pass
       
        
    print("edges:",edge_points_y)
    print("check points",test_point_y)
print("----------")
print("interior points",interior_points_y)

filtered_interior_y = [point for point in interior_points_y if point not in polygon]
print(filtered_interior_y)


# Unzip coordinates for plotting
x, y = zip(*polygon)
c, d = zip(*points)
a, b = zip(*filtered_interior_x)
g ,h = zip(*filtered_interior_y)

# Plotting
plt.plot(x, y, 'r-', linewidth=2, label='Polygon Boundary')  # Red line
plt.scatter(x, y, color='red', s=50, label='Vertices')       # Red markers
plt.scatter(c, d, color ="blue", s= 5)
plt.scatter(a, b, color = "black", s = 5)       # x-marker
plt.scatter(g, h, color = "black", s = 5)       # y-marker
plt.fill(x, y, alpha=0.2, color='red')                       # Light red fill

# Annotations and labels
plt.title("Polygon Visualization with Red Markers", fontsize=14)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.axis('equal')  # Fix aspect ratio

plt.show()