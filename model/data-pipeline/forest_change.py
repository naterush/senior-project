from sklearn import linear_model, tree
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import rasterio
import numpy as np
import PIL
import math
import pprint
import warnings
import pickle
import time
from random import randint
import xmltodict


def get_prediction_map(model_filepath, satellite_jpg_filepath, metadata_filepath, conus_data_filepath, pixel_radius=4):
    # Return a grid of predictions
    pass


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
    return (ul_x, ul_y, lr_x, lr_y)


def predict_change_two_images(im1_path, img2_path, img1_metadata, img2_metadata, conus_img, model_fp=None):

    # ACTUAL CODE FOR GETTING THE PREDICTIONS
    # pred_map_1 = get_prediction_map(model_fp, img1_path, img1_metadata, conus_img)
    # pred_map_2 = get_prediction_map(model_fp, img2_path, img2_metadata, conus_img)

    # TESTING CODE FOR GENERATING FAKE PREDICTION DATA
    ds = rasterio.open(forest_nonforest_img)
    band1 = ds.read(1)
    pred_map_1 = band1[4000:4600, 1000:1600].copy()
    pred_map_2 = pred_map_1.copy()
    # Alter region2 to simulate forest destruction
    pct = 80 # pct/100 of pred_map_2 will be 'destroyed'
    for y in range(400, 500):
        for x in range(50, 150):
            if randint(1, 100) < pct:
                pred_map_2[y, x] = 2

    pass


def find_changes(region1, region2):
    pass


model_fp = 'model/matching_coordinates/logreg_model_3class.sav'
conus_fp = 'model/data-pipeline/conus_forest_nonforest.img'
jpg_before = 'model/matching_coordinates/case_study_data/walker_fire_before1.jpg'
metadata_before = 'model/matching_coordinates/case_study_data/walker_fire_before1.xml'
jpg_after = 'model/matching_coordinates/case_study_data/walker_fire_after1.jpg'
metadata_after = 'model/matching_coordinates/case_study_data/walker_fire_after1.xml'

predict_change_two_images(jpg_before, jpg_after, metadata_before, metadata_after, conus_fp, model_fp)


# IN THIS PHOTO: Black is forest (1), Grey is nonforest (2), White is water (3)
plt.imshow(pred_map_1, cmap = "gray")
plt.imshow(pred_map_2-pred_map_1, cmap = "gray")




ds = rasterio.open('model/data-pipeline/conus_forest_nonforest.img')
band1 = ds.read(1)
pred_map_1 = band1[4000:4600, 1000:1600].copy()
pred_map_2 = pred_map_1.copy()
# Alter region2 to simulate forest destruction
pct = 20 # pct/100 of pred_map_2 will be 'destroyed'
for y in range(400, 500):
    for x in range(50, 150):
        if randint(1, 100) < pct:
            pred_map_2[y, x] = 2
change_map = pred_map_2
