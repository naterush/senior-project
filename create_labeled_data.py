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
import math
import pprint


# This function generates [x_coord, y_coord, averageRed, averageGreen, averageBlue, forestCoverLabel]
#  instances for the given satellite image
# The bounds of the satellite image are given in Albers Equal Area form, which can
#      be found in the metadata for each downloaded image
def generate_training_data(satellite_jpg_filepath, conus_data_filepath, ul_x, ul_y, lr_x, lr_y, pixel_radius=5):

    # Open the satellite image
    img = PIL.Image.open(jpg_filepath)
    # TODO: Check for errors in opening
    rgb_data = np.asarray(img)
    jpg_width = rgb_data.shape[0]
    jpg_height = rgb_data.shape[1]

    # Open the Conus/Non-Conus Dataset
    ds = rasterio.open(forest_nonforest_img)
    band1 = ds.read(1)
    plt.imshow(band1, cmap = "gray")

    # For each labeled point in the Conus dataset within the JPG image, create entry
    start_x = int((ul_x - ds.bounds.left)//250)
    end_x = int(start_x + ((lr_x - ul_x)//250))
    start_y = int((ds.bounds.top - ul_y)//250)
    end_y = int(start_y + ((ul_y - lr_y)//250))

    forest_cover = band1[start_y:end_y, start_x:end_x].copy()
    # forest_image = np.zeros(shape=(600, 600, 3))

    # plt.imshow(forest_cover, cmap = "gray")

    results = []
    for x in range(0, len(forest_cover[0])):
        # Debugging
        if x % 10 == 0: print('On column ' + str(x) + '/600 of image')

        for y in range(0, len(forest_cover)):
            # Get the appropriate coordinates within the JPG image
            jpg_x = int(((x / (len(forest_cover[0])))*jpg_width))
            jpg_y = int(((y / (len(forest_cover)))*jpg_height))

            avgR = np.average(rgb_data[jpg_y-pixel_radius:jpg_y+pixel_radius,
                                       jpg_x-pixel_radius:jpg_x+pixel_radius,
                                       0])
            avgG = np.average(rgb_data[jpg_y-pixel_radius:jpg_y+pixel_radius,
                                       jpg_x-pixel_radius:jpg_x+pixel_radius,
                                       1])
            avgB = np.average(rgb_data[jpg_y-pixel_radius:jpg_y+pixel_radius,
                                       jpg_x-pixel_radius:jpg_x+pixel_radius,
                                       2])
            # forest_image[y, x] = [int(avgR) if not math.isnan(avgR) else 0,
            #                       int(avgG) if not math.isnan(avgG) else 0,
            #                       int(avgB) if not math.isnan(avgB) else 0]

            albers_x = ds.bounds.left + (250*(start_x+x))
            albers_y = ds.bounds.top - (250*(start_y+y))
            forest_label = forest_cover[y, x]

            results.append([albers_x, albers_y, avgR, avgG, avgB, forest_label])


    df = pd.DataFrame(columns=['albers_x', 'albers_y', 'avg_red', 'avg_green', 'avg_blue', 'forest_label'],
                      data = results)
    return df
    # return (df, forest_image)


forest_nonforest_img = '/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/matching_coordinates/conus_forest_nonforest.img'
xml_metadata = 'model/matching_coordinates/sample_data/LT05_CU_028008_20080822_20181220_C01_V01.xml'
jpg_filepath = 'model/matching_coordinates/sample_data/LT05_CU_028008_20080822_20181220_C01_V01.jpg'
# Get the bounding coordinates
fd = open(xml_metadata)
metadata = xmltodict.parse(fd.read())
fd.close()

# Get the Albers Equal Area bounds of the satellite image
projection_bounds = metadata['ard_metadata']['tile_metadata']['global_metadata']['projection_information']
left_x = float(projection_bounds['corner_point'][0]['@x'])
right_x = float(projection_bounds['corner_point'][1]['@x'])
top_y = float(projection_bounds['corner_point'][0]['@y'])
bottom_y = float(projection_bounds['corner_point'][1]['@y'])

(df, forest_image) = generate_training_data(jpg_filepath, forest_nonforest_img, left_x, top_y, right_x, bottom_y)
plt.imshow(forest_image, cmap = "gray")
display(df.drop_na())
