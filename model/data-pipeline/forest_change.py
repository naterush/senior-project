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

# ds = rasterio.open(forest_nonforest_img)
# band1 = ds.read(1)
# plt.imshow(band1, cmap='gray')

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

def create_fake_regions():
    ds = rasterio.open('model/data-pipeline/conus_forest_nonforest.img')
    band1 = ds.read(1)
    pred_map_1 = band1[4000:4600, 1000:1600].copy()
    pred_map_2 = pred_map_1.copy()
    # ADD FOREST DESCTRUCTION
    pct = 50 # pct/100 of pred_map_2 will be 'destroyed'
    for y in range(400, 500):
        for x in range(50, 150):
            if randint(1, 100) < pct:
                pred_map_2[y, x] = 2
    for y in range(100, 200):
        for x in range(100, 200):
            if randint(1, 100) < pct:
                pred_map_2[y, x] = 2
    # ADD FOREST GROWTH
    for y in range(300, 400):
        for x in range(400, 500):
            if randint(1, 100) < pct:
                pred_map_2[y, x] = 1
    return (pred_map_1, pred_map_2)

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




def find_changes(region1, region2, grid_size=60, thresh_pct = 0.20):
    change_map = np.zeros(region1.shape, dtype=int)
    height = region1.shape[0]
    width = region1.shape[1]
    # pixels_per_square = (height//grid_size) * (width//grid_size)
    thresh = 0
    for y_top_left in range(0, height, height//grid_size):
        for x_top_left in range(0, width, width//grid_size):
            r1_square = region1[y_top_left:y_top_left+grid_size,
                                        x_top_left:x_top_left+grid_size]
            r2_square = region2[y_top_left:y_top_left+grid_size,
                                        x_top_left:x_top_left+grid_size]
            change_map_region = change_map[y_top_left:y_top_left+grid_size,
                                        x_top_left:x_top_left+grid_size]
            # Get only the pixels predictions changed
            s1 = np.sum(r1_square)
            s2 = np.sum(r2_square)
            # TODO: Add some threshold of change required to flag it
            if s1 < s2 and (s2-s1 > thresh):
                # There has been a LOSS in forest cover
                change_map_region[:, :] = 1
            elif s2 < s1 and (s1 - s2 > thresh):
                # There has been an GAIN in forest cover
                change_map_region[:, :] = 2
            else:
                # NO CHANGE
                change_map_region[:, :] = 0
    return change_map

(pred_map_1, pred_map_2) = create_fake_regions()
c = find_changes(pred_map_1, pred_map_2)
plt.imshow(pred_map_1, cmap = "gray")
plt.imshow(pred_map_2, cmap = "gray")

plt.imshow(c, cmap = "gray")

model_fp = 'model/matching_coordinates/logreg_model_3class.sav'
conus_fp = 'model/data-pipeline/conus_forest_nonforest.img'
jpg_before = 'model/matching_coordinates/case_study_data/walker_fire_before1.jpg'
metadata_before = 'model/matching_coordinates/case_study_data/walker_fire_before1.xml'
jpg_after = 'model/matching_coordinates/case_study_data/walker_fire_after1.jpg'
metadata_after = 'model/matching_coordinates/case_study_data/walker_fire_after1.xml'

predict_change_two_images(jpg_before, jpg_after, metadata_before, metadata_after, conus_fp, model_fp)


# IN THIS PHOTO: Black is forest (1), Grey is nonforest (2), White is water (3)
plt.imshow(pred_map_1, cmap = "gray")
plt.imshow(pred_map_2, cmap = "gray")





print(len(change_map[change_map == 2]))
