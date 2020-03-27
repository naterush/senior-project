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
import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json
import requests
import xmltodict
import numpy as np
import PIL
import pprint

username = 'ejperelmuter'
password = 'Sapling#2020'

landsat_api = landsatxplore.api.API(username, password)
print(landsat_api)

# TODO: Get an optical image from USGS
# TODO: Produce [Lat, Long, R, G, B] from it

xml_metadata = 'model/matching_coordinates/sample_data/LT05_CU_028008_20080822_20181220_C01_V01.xml'
jpg_filepath = 'model/matching_coordinates/sample_data/LT05_CU_028008_20080822_20181220_C01_V01.jpg'
# Get the bounding coordinates
fd = open(xml_metadata)
metadata = xmltodict.parse(fd.read())
fd.close()

# bounds = metadata['ard_metadata']['tile_metadata']['global_metadata']['bounding_coordinates']
# north = float(bounds['north'])
# south = float(bounds['south'])
# east = float(bounds['east'])
# west = float(bounds['west'])
# pprint.pprint(projection_bounds)

projection_bounds = metadata['ard_metadata']['tile_metadata']['global_metadata']['projection_information']
left = float(projection_bounds['corner_point'][0]['@x'])
right = float(projection_bounds['corner_point'][1]['@x'])
top = float(projection_bounds['corner_point'][0]['@y'])
bottom = float(projection_bounds['corner_point'][1]['@y'])
units_per_pixel = (top-bottom)/5000

# https://spatialreference.org/ref/epsg/wgs-84/ --> EPSG 4326

# Open the satellite image
img = PIL.Image.open(jpg_filepath)
rgb_data = np.asarray(img)
width = rgb_data.shape[0]
height = rgb_data.shape[1]

# Open the Conus/Non-Conus Dataset
forest_nonforest_img = '/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/matching_coordinates/conus_forest_nonforest.img'
ds = rasterio.open(forest_nonforest_img)
band1 = ds.read(1)
plt.imshow(band1, cmap = "gray")

# For each point on band1 within the JPG image, create entry
# Data Generated: (conus_x, conus_y, avg_r, avg_g, avb_b, forest_boolean)

start_x = int((left - ds.bounds.left)//250)
end_x = int(start_x + ((right - left)/250))
start_y = int((ds.bounds.top - top)//250)
end_y = int(start_y + ((top - bottom)/250))

# TESTING
end_x = start_x + 10
end_y = start_y + 10
arr = []

for x in range(start_x, end_x):
    for y in range(start_y, end_y):
        is_forest = band1[y][x]
        aea_x = ds.bounds.left + (250*x)
        aea_y = ds.bounds.top - (250*y)
        # Get closest pixel
        jpg_x = (aea_x - left)//units_per_pixel
        jpg_y = (top - aea_y)//units_per_pixel

        arr.append(is_forest)

print(len((arr)))
