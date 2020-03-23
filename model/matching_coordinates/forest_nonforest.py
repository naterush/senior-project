from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import pandas as pd
import math
from math import sqrt
from pyproj import Proj, transform



def coord_to_lat_long(x, y):
    phi_0 = 23.000000 #  latitude for the origin of the Cartesian coordinates
    lambda_0 =  -96.000000 #  longitude for the origin of the Cartesian coordinates

    phi_0 = 51.652084
    lambda_0 = -127.977889

    phi_1 = 29.50
    phi_2 = 45.50

    phi_0 = math.radians(phi_0)
    lambda_0 = math.radians(lambda_0)
    phi_1 = math.radians(phi_1)
    phi_2 = math.radians(phi_2)

    n = (1/2)*(math.sin(phi_1) + math.sin(phi_2))
    C = (math.cos(phi_1)**2) + (2*n*math.sin(phi_1))
    p_0 = (sqrt(C - (2 * n * math.sin(phi_0)))) / n

    p = sqrt((x)**2 + (p_0-y)**2)
    theta = math.atan(x/(p_0-y))

    x = (C - ((p**2)*(n**2)))/(2*n)
    print(x)
    calculated_lat = math.asin(x)
    calculated_long = lambda_0 + (theta/n)
    a = math.degrees(calculated_lat)
    b = math.degrees(calculated_long)
    return (a, b)
    # return (calculated_lat, calculated_long)
# l = coord_to_lat_long(1, 1)


# read in the image
forest_nonforest_img = '/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/matching_coordinates/conus_forest_nonforest.img'
ds = rasterio.open(forest_nonforest_img)

print("Dataset Name: " + ds.name)
# print("Dataset Mode: " + ds.mode)
# print("Band Count: " + str(ds.count))
width = ds.width
height = ds.height
print("Dataset Width: " + str(width))
print("Dataset Height: " + str(height))
print("Dataset Bounds: ", ds.bounds)
# print("Dataset Transform: ", ds.transform)
# ul = ds.transform * (0, 0)
# print("Upper Left Corner: ", ul)
# lr = ds.transform * (ds.width, ds.height)
# print("Lower Right Corner: ", lr)


# band1 contains the biomass data we are interested in
band1 = ds.read(1)
plt.imshow(band1, cmap = "gray")

# a = 10384
# b = 15480
# florida = band1[a:a+40,b:b+40]
# plt.imshow(florida, cmap = "gray", aspect='auto')

# y = top - (a*250)
# print(y)
# x = left + (b*250)
# print(x)


data = band1
print(data)


left = -2361625
right = 2263375
bottom = 262875
top = 3177625

h = top - bottom
print(h)
print(h/250)
# print((top - bottom) / 250)
# print(height)
# print((right - left)/250)
# print(width)


# x2,y2 = transform(inProj,outProj,x1,y1)

# curr_x = -2229109 # Should iterate --> 2145923.51
# curr_y = 3556313

height = len(data)
width = len(data[0])



inProj = Proj(init='epsg:5070')
outProj = Proj(init='epsg:4269')
curr_y = top

for y in range(0, 10):
    # curr_x = -2229109 # Far left (west) coordinate
    curr_x = left
    if y % 1000 == 0:
        print('y: ' + str(y))

    for x in range(0, 1):
        # x1 = left + (x*250)
        # y1 = top - (250*y)
        # x2,y2 = transform(inProj,outProj,x1,y1)

        # curr_x and curr_y hold the correct EPSG 5070 x-y coordinates
        long, lat = transform(inProj,outProj,curr_x,curr_y)
        print('('+str(lat)+', ' + str(long)+')')

        curr_x = curr_x + 250

    curr_y = curr_y - 250
