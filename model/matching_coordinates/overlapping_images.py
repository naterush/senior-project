import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import rasterio
import pandas as pd
from pyproj import Proj, transform
import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json
import requests
import xmltodict
import numpy as np
import PIL
import pprint


# TODO: Get an optical image from USGS
# TODO: Produce [Lat, Long, R, G, B] from it
xml_metadata = 'model/matching_coordinates/sample_data/LT05_CU_028008_20080822_20181220_C01_V01.xml'
jpg_filepath = 'model/matching_coordinates/sample_data/LT05_CU_028008_20080822_20181220_C01_V01.jpg'

# Get the bounding coordinates
fd = open(xml_metadata)
metadata = xmltodict.parse(fd.read())
fd.close()


# Open the satellite image
img = PIL.Image.open(jpg_filepath)
rgb_data = np.asarray(img)
width = rgb_data.shape[0]
height = rgb_data.shape[1]

# Get the Albers Equal Area bounds of the satellite image

projection_bounds = metadata['ard_metadata']['tile_metadata']['global_metadata']['projection_information']
print(projection_bounds)
left_x = float(projection_bounds['corner_point'][0]['@x'])
right_x = float(projection_bounds['corner_point'][1]['@x'])
top_y = float(projection_bounds['corner_point'][0]['@y'])
bottom_y = float(projection_bounds['corner_point'][1]['@y'])
meters_per_pixel = (top_y-bottom_y)/height
print(meters_per_pixel)


# Open the Conus/Non-Conus Dataset
forest_nonforest_img = '/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/matching_coordinates/conus_forest_nonforest.img'
ds = rasterio.open(forest_nonforest_img)
band1 = ds.read(1)
plt.imshow(band1, cmap = "gray")

# For each point on band1 within the JPG image, create entry
# Data Generated: (conus_x, conus_y, avg_r, avg_g, avb_b, forest_boolean)

start_x = int((left_x - ds.bounds.left)//250)
end_x = int(start_x + ((right_x - left_x)//250))
start_y = int((ds.bounds.top - top_y)//250)
end_y = int(start_y + ((top_y - bottom_y)//250))

forest_cover = band1[start_y:end_y, start_x:end_x]
plt.imshow(forest_cover, cmap = "gray")

test_jpg = rgb_data.copy()
plt.imshow(test_jpg)

jpg_width = width
jpg_height = height
for x in range(0, len(forest_cover[0])):
    for y in range(0, len(forest_cover)):
        jpg_x = ((x / (len(forest_cover[0])))*jpg_width)
        jpg_y = ((y / (len(forest_cover)))*jpg_height)
        # test_jpg[int(jpg_y), int(jpg_x)] = 255

        is_forest = forest_cover[y, x]
        # 3: Water, 2: Some cover,
        if is_forest == 3:
            test_jpg[int(jpg_y)-2:int(jpg_y)+2, int(jpg_x)-2:int(jpg_x)+2] = 255
            # test_jpg[int(jpg_y), int(jpg_x)] = 255

plt.imshow(test_jpg)
