# CALIFORNIA FIRE CASE STUDY
# This script trains a model on 3 images from california, then predicts a 4th image
#

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
import cv2
warnings.filterwarnings("ignore")


def brighten_image(rgb_img_data, value = 30):
    hsv = cv2.cvtColor(rgb_img_data, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    brightened_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return brightened_image


# Input: Takes in a JPG image with bounds given in an XML file, conus/non-conus map
# Produces a np array of training data for only the colored portions of the image
# Example instance produced in data: [averageR, averageG, averageB, conusLabel]
def get_labeled_data(satellite_jpg_filepath, metadata_xml_filepath, conus_data_filepath, pixel_radius=5):

    # Get the bounding coordinates from the metadata file
    fd = open(metadata_xml_filepath)
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

    max_x = len(forest_cover[0])
    max_y = len(forest_cover)
    results = np.zeros((max_x*max_y, 4))
    row_num = 0
    for x in range(0, max_x):
        if x % 50 == 0: print('On column ' + str(x) + '/'+str(max_x)+' of image')
        for y in range(0, max_y):
            # 1 = Forest Cover, 2 = No Forest, 3 = Water
            # 0 = No Label (for off coast points in dataset, should be ignored as a label)
            forest_label = forest_cover[y, x]
            if forest_label == 0:
                continue

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
                albers_x = ds.bounds.left + (250*(start_x+x))
                albers_y = ds.bounds.top - (250*(start_y+y))

                results[row_num, 0] = avgR
                results[row_num, 1] = avgG
                results[row_num, 2] = avgB
                results[row_num, 3] = forest_label
                row_num = row_num + 1

    all_results = results[:row_num].copy()
    print("Black percentage: " + str(((max_x*max_y) - row_num)/(max_x*max_y)))
    return all_results

def create_prediction_map(model, satellite_jpg_filepath, metadata_xml_filepath, conus_data_filepath, pixel_radius=5):
    # Get the bounding coordinates from the metadata file
    fd = open(metadata_xml_filepath)
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
    prediction_map = np.zeros(forest_cover.shape, dtype=np.uint8)
    prediction_map[:, :] = 3

    max_x = len(forest_cover[0])
    max_y = len(forest_cover)
    # results = np.zeros((max_x*max_y, 4))
    row_num = 0
    preds = []
    actual_labels = []
    for x in range(0, max_x):
        if x % 100 == 0: print('On column ' + str(x) + '/'+str(max_x)+' of image')
        for y in range(0, max_y):
            forest_label = forest_cover[y, x]
            if forest_label == 0:
                continue

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
                albers_x = ds.bounds.left + (250*(start_x+x))
                albers_y = ds.bounds.top - (250*(start_y+y))

                # Predict here
                pred = model.predict([[avgR, avgG, avgB]])
                prediction_map[y, x] = int(pred[0])
                preds.append(pred[0])
                actual_labels.append(forest_label)

                row_num = row_num + 1

    # all_results = results[:row_num].copy()
    print("Black percentage: " + str(((max_x*max_y) - row_num)/(max_x*max_y)))
    return (forest_cover, prediction_map, preds, actual_labels)

def get_bounds(metadata_filepath):
    # Get the bounding coordinates from the metadata file
    if metadata_filepath.endswith(".xml"):
        with open(metadata_filepath) as fd:
            metadata = xmltodict.parse(fd.read())

        # Get the Albers Equal Area bounds of the satellite image
        projection_bounds = metadata['ard_metadata']['tile_metadata']['global_metadata']['projection_information']
        ul_x = float(projection_bounds['corner_point'][0]['@x'])
        ul_y = float(projection_bounds['corner_point'][0]['@y'])
        lr_x = float(projection_bounds['corner_point'][1]['@x'])
        lr_y = float(projection_bounds['corner_point'][1]['@y'])
    elif metadata_filepath.endswith(".txt"):
        with open(metadata_filepath) as fd:
            lines = [l.strip() for l in fd.readlines()]

        ul_line = [l for l in lines if l.startswith("UL_CORNER")][0].split("=")[1].strip()[1:-1]
        lr_line = [l for l in lines if l.startswith("LR_CORNER")][0].split("=")[1].strip()[1:-1]
        ul_x = float(ul_line.split(",")[0])
        ul_y = float(ul_line.split(",")[1])
        lr_x = float(lr_line.split(",")[0])
        lr_y = float(lr_line.split(",")[1])
        # TODO: ZONE HERE
        utm_zone = int([l for l in lines if l.startswith("UTM_ZONE")][0].split("=")[1].strip())
        # print("got zone", utm_zone)
        inProj = Proj(proj="utm",zone=utm_zone,ellps="WGS84", south=False)
        outProj = Proj(init='epsg:5070')

        (ul_x, ul_y) = transform(inProj,outProj,ul_x,ul_y)
        (lr_x, lr_y) = transform(inProj,outProj,lr_x,lr_y)

    return (ul_x, ul_y, lr_x, lr_y)

def get_prediction_map(model_filepath, satellite_jpg_filepath, metadata_filepath, conus_data_filepath, pixel_radius=4):
    loaded_model = pickle.load(open(model_filepath, 'rb'))

    (ul_x, ul_y, lr_x, lr_y) = get_bounds(metadata_filepath)

    # Open the satellite image
    img = PIL.Image.open(satellite_jpg_filepath)
    rgb_data = np.asarray(img)
    jpg_width = rgb_data.shape[1]
    jpg_height = rgb_data.shape[0]

    # Open the Conus/Non-Conus Dataset
    ds = rasterio.open(conus_data_filepath)
    # band1 = ds.read(1)

    # For each labeled point in the Conus dataset within the JPG image, create entry
    start_x = int((ul_x - ds.bounds.left)//250)
    end_x = int(start_x + ((lr_x - ul_x)//250))
    start_y = int((ds.bounds.top - ul_y)//250)
    end_y = int(start_y + ((ul_y - lr_y)//250))

    # forest_cover = band1[start_y:end_y, start_x:end_x].copy()
    # prediction_map = forest_cover.copy()
    prediction_map = np.zeros((end_y-start_y, end_x-start_x), dtype=np.uint8)
    # print(f'FC Shape: {forest_cover.shape}   PredMap Shape: {prediction_map.shape}')
    # prediction_map[:, :] = 0

    max_x = len(prediction_map[0])
    max_y = len(prediction_map)
    row_num = 0
    for x in range(0, max_x):
        if x % 100 == 0: print(str(x) + '/'+str(max_x)+'...', end =" ")
        for y in range(0, max_y):
            # Get the appropriate coordinates within the JPG image
            jpg_x = int(((x / max_x)*jpg_width))
            jpg_y = int(((y / max_y)*jpg_height))
            pixel = rgb_data[jpg_y, jpg_x]
            if not (pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0):
                pixel_square = rgb_data[jpg_y-pixel_radius:jpg_y+pixel_radius,
                                 jpg_x-pixel_radius:jpg_x+pixel_radius]

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
                row_num = row_num + 1 # Track how full the photo is
    return prediction_map

# Train on cali1-3 images, then run predictions on adam's
forest_nonforest_img = 'model/matching_coordinates/conus_forest_nonforest.img'
logreg_fp = 'model/matching_coordinates/logreg_model_3class.sav'
dtree_fp = 'server/dtree_3class.sav'

before_img = 'model/matching_coordinates/case_study_data/walker_fire_before1.jpg'
before_meta = 'model/matching_coordinates/case_study_data/walker_fire_before1.xml'

after_img = 'model/matching_coordinates/case_study_data/walker_fire_after2.jpg'
after_meta = 'model/matching_coordinates/case_study_data/walker_fire_after2.xml'

before_pred = get_prediction_map(logreg_fp, before_img, before_meta, forest_nonforest_img)
plt.imshow(before_pred, cmap='gray')
