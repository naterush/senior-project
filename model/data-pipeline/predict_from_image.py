from sklearn import linear_model, tree
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import rasterio
import pandas as pd
import xmltodict
import numpy as np
import PIL
import math
import pprint
import warnings
import pickle
import time
warnings.filterwarnings("ignore")

# Input: Takes in a JPG image with bounds given in an XML file, conus/non-conus map
# Produces a np array of training data for only the colored portions of the image
# Example instance produced in data: [averageR, averageG, averageB, conusLabel]
def get_prediction_map(model_filepath, satellite_jpg_filepath, metadata_filepath, conus_data_filepath, pixel_radius=5):
    loaded_model = pickle.load(open(model_filepath, 'rb'))
    # Get the bounding coordinates from the metadata file
    fd = open(metadata_filepath)
    metadata = xmltodict.parse(fd.read())
    fd.close()

    # Get the Albers Equal Area bounds of the satellite image
    projection_bounds = metadata['ard_metadata']['tile_metadata']['global_metadata']['projection_information']
    ul_x = float(projection_bounds['corner_point'][0]['@x'])
    ul_y = float(projection_bounds['corner_point'][0]['@y'])
    lr_x = float(projection_bounds['corner_point'][1]['@x'])
    lr_y = float(projection_bounds['corner_point'][1]['@y'])

    # Open the satellite image
    img = PIL.Image.open(satellite_jpg_filepath)
    # TODO: Check for errors in opening
    rgb_data = np.asarray(img)
    jpg_width = rgb_data.shape[1]
    jpg_height = rgb_data.shape[0]

    # Open the Conus/Non-Conus Dataset
    ds = rasterio.open(conus_data_filepath)
    band1 = ds.read(1)

    # For each labeled point in the Conus dataset within the JPG image, create entry
    start_x = int((ul_x - ds.bounds.left)//250)
    end_x = int(start_x + ((lr_x - ul_x)//250))
    start_y = int((ds.bounds.top - ul_y)//250)
    end_y = int(start_y + ((ul_y - lr_y)//250))

    forest_cover = band1[start_y:end_y, start_x:end_x].copy()
    prediction_map = forest_cover.copy()
    prediction_map[:, :] = 0

    max_x = len(forest_cover[0])
    max_y = len(forest_cover)
    row_num = 0
    for x in range(0, max_x):
        if x % 100 == 0: print('On column ' + str(x) + '/'+str(max_x)+' of image')
        for y in range(0, max_y):
            # 1 = Forest Cover, 2 = No Forest, 3 = Water
            # 0 = No Label (for off coast points in dataset, should be ignored as a label)
            # forest_label = forest_cover[y, x]
            # if forest_label == 0:
            #     continue

            # Get the appropriate coordinates within the JPG image
            jpg_x = int(((x / max_x)*jpg_width))
            jpg_y = int(((y / max_y)*jpg_height))
            pixel = rgb_data[jpg_y, jpg_x]
            if pixel[0] != 0 and pixel[1] != 0 and pixel[2] != 0:
                pixel_square = rgb_data[jpg_y-pixel_radius:jpg_y+pixel_radius,
                                 jpg_x-pixel_radius:jpg_x+pixel_radius]
                # print(f"Calculating average on {slice.size, slice.shape}")
                avgR = np.average(pixel_square[:, :, 0])
                if np.isnan(avgR):
                    # Move onto next pixel if there are empty pixels in this radius
                    continue
                avgG = np.average(pixel_square[:, :, 1])
                avgB = np.average(pixel_square[:, :, 2])
                # albers_x = ds.bounds.left + (250*(start_x+x))
                # albers_y = ds.bounds.top - (250*(start_y+y))
                pred = loaded_model.predict([[avgR, avgG, avgB]])
                prediction_map[y, x] = pred[0]
                # prediction_map[y, x] = 2 if pred[0] == 0 else 1

                row_num = row_num + 1 # Track how full the photo is

    print("Black percentage: " + str(((max_x*max_y) - row_num)/(max_x*max_y)))
    return prediction_map

# r_band = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/LC08_L1TP_014032_20200316_20200326_01_T1_B4.TIF'
# g_band = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/LC08_L1TP_014032_20200316_20200326_01_T1_B3.TIF'
# b_band = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/LC08_L1TP_014032_20200316_20200326_01_T1_B2.TIF'

# TESTING WITH A SINGLE IMAGE
# model_fp = 'model/matching_coordinates/logreg_model_3class.sav'
# jpg = 'model/matching_coordinates/sample_data/adam_cali.jpg'
# metadata = 'model/matching_coordinates/sample_data/adam_cali.xml'
# conus_fp = 'model/data-pipeline/conus_forest_nonforest.img'
# pred_map4 = get_prediction_map(model_fp, jpg, metadata, conus_fp, pixel_radius=4)


model_fp = 'model/matching_coordinates/logreg_model_3class.sav'
model_fp = 'model/matching_coordinates/dtree_3class.sav'
conus_fp = 'model/data-pipeline/conus_forest_nonforest.img'
jpg_before = 'model/matching_coordinates/case_study_data/walker_fire_before1.jpg'
metadata_before = 'model/matching_coordinates/case_study_data/walker_fire_before1.xml'
jpg_after = 'model/matching_coordinates/case_study_data/walker_fire_after1.jpg'
metadata_after = 'model/matching_coordinates/case_study_data/walker_fire_after1.xml'

s = time.time()
pred_map_before = get_prediction_map(model_fp, jpg_before, metadata_before, conus_fp, pixel_radius=4)
print(str(round((time.time() - s), 2)) + ' seconds for BEFORE photo')

s = time.time()
pred_map_after = get_prediction_map(model_fp, jpg_after, metadata_after, conus_fp, pixel_radius=4)
print(str(round((time.time() - s), 2)) + ' seconds for AFTER photo')



plt.imshow(pred_map_before, cmap='gray')
plt.imshow(pred_map_after, cmap='gray')
# # print(pred_map_after[500, 100:])
# if True:
#     crop = pred_map_before[300:500, 100:400]
#     # crop = pred_map_after[300:500, 100:400]
#     plt.imshow(crop, cmap='gray')
#
# if True:
#     crop = pred_map_after[300:500, 100:400]
#     plt.imshow(crop, cmap='gray')
