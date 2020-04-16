import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp
import pandas as pd

from pyproj import Proj, transform


# read in the image
forest_nonforest_img = 'conus_forest_nonforest.img'
ds = rasterio.open(forest_nonforest_img)

print("Dataset Name: " + ds.name)
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

exit(1)

a = 10384
b = 15480
florida = band1[a:a+40,b:b+40]
plt.imshow(florida, cmap = "gray", aspect='auto')

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
