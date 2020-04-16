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
    results = np.zeros((max_x*max_y, 4))
    row_num = 0
    for x in range(0, max_x):
        if x % 50 == 0: print('On column ' + str(x) + '/'+str(max_x)+' of image')
        for y in range(0, max_y):
            # 1 = Forest Cover, 2 = No Forest, 3 = Water
            # 0 = No Label (for off coast points in dataset, should be ignored as a label)
            # forest_label = forest_cover[y, x]
            # if forest_label == 0:
            #     continue

            # Get the appropriate coordinates within the JPG image
            jpg_x = int(((x / max_x)*jpg_width))
            jpg_y = int(((y / max_y)*jpg_height))

            if np.count_nonzero(rgb_data[jpg_y, jpg_x]) != 0:
                slice = rgb_data[jpg_y-pixel_radius:jpg_y+pixel_radius,
                                 jpg_x-pixel_radius:jpg_x+pixel_radius]
                # print(f"Calculating average on {slice.size, slice.shape}")
                avgR = np.average(slice[:, :, 0])
                if np.isnan(avgR):
                    # Move onto next pixel if there are empty pixels in this radius
                    continue
                avgG = np.average(slice[:, :, 1])
                avgB = np.average(slice[:, :, 2])
                # albers_x = ds.bounds.left + (250*(start_x+x))
                # albers_y = ds.bounds.top - (250*(start_y+y))

                results[row_num, 0] = avgR
                results[row_num, 1] = avgG
                results[row_num, 2] = avgB
                pred = loaded_model.predict([[avgR, avgG, avgB]])
                prediction_map[y, x] = pred[0]
            # else:
                # prediction_map[y, x] = pred

    # all_results = results[:row_num].copy()
    print("Black percentage: " + str(((max_x*max_y) - row_num)/(max_x*max_y)))
    return prediction_map

r_band = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/LC08_L1TP_014032_20200316_20200326_01_T1_B4.TIF'
g_band = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/LC08_L1TP_014032_20200316_20200326_01_T1_B3.TIF'
b_band = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/LC08_L1TP_014032_20200316_20200326_01_T1_B2.TIF'

model_fp = 'model/matching_coordinates/logreg_model_2class.sav'
jpg = 'model/matching_coordinates/sample_data/adam_cali.jpg'
metadata = 'model/matching_coordinates/sample_data/adam_cali.xml'
conus_fp = 'model/data-pipeline/conus_forest_nonforest.img'

pred_map = get_prediction_map(model_fp, jpg, metadata, conus_fp, pixel_radius=5)

plt.imshow(jpg)
plt.imshow(pred_map, cmap='gray')
