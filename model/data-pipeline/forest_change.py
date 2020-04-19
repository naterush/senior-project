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
from pyproj import Proj, transform
import xmltodict
warnings.filterwarnings("ignore")

def get_prediction_map(model_filepath, satellite_jpg_filepath, metadata_filepath, conus_data_filepath, pixel_radius=4):
    loaded_model = pickle.load(open(model_filepath, 'rb'))

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
    prediction_map = forest_cover.copy()
    prediction_map[:, :] = 0

    max_x = len(forest_cover[0])
    max_y = len(forest_cover)
    row_num = 0
    for x in range(0, max_x):
        if x % 100 == 0: print('On column ' + str(x) + '/' + str(max_x) + ' of image')
        for y in range(0, max_y):
            # Get the appropriate coordinates within the JPG image
            jpg_x = int(((x / max_x)*jpg_width))
            jpg_y = int(((y / max_y)*jpg_height))
            pixel = rgb_data[jpg_y, jpg_x]
            if pixel[0] != 0 and pixel[1] != 0 and pixel[2] != 0:
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

    print("Black percentage: " + str(((max_x*max_y) - row_num)/(max_x*max_y)))
    return prediction_map


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


def find_changes(region1, region2, grid_size=60, thresh_pct = 0):
    change_map = np.zeros(region1.shape, dtype=int)
    height = region1.shape[0]
    width = region1.shape[1]
    pixels_per_square = (height//grid_size) * (width//grid_size)
    print(pixels_per_square)
    thresh = pixels_per_square * 0.50
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
            if (np.count_nonzero(r1_square) <= thresh) or (np.count_nonzero(r2_square) <= thresh):
                change_map_region[:, :] = 0
                continue
            # Ignore if either square is mostly black
            if s1 <= pixels_per_square or s2 <= pixels_per_square:
                change_map_region[:, :] = 0
            # TODO: Add some threshold of change required to flag it
            elif s1 < s2 and (s2-s1 > thresh):
                # There has been a LOSS in forest cover
                change_map_region[:, :] = 1
            elif s2 < s1 and (s1 - s2 > thresh):
                # There has been an GAIN in forest cover
                change_map_region[:, :] = 2
            else:
                # NO CHANGE
                change_map_region[:, :] = 0
    return change_map

model_fp = 'model/matching_coordinates/logreg_model_3class.sav'
conus_fp = 'model/data-pipeline/conus_forest_nonforest.img'
jpg_before = 'model/matching_coordinates/case_study_data/walker_fire_before2.jpg'
metadata_before = 'model/matching_coordinates/case_study_data/walker_fire_before2.xml'
jpg_after = 'model/matching_coordinates/case_study_data/walker_fire_after1.jpg'
metadata_after = 'model/matching_coordinates/case_study_data/walker_fire_after1.xml'

# predict_change_two_images(jpg_before, jpg_after, metadata_before, metadata_after, conus_fp, model_fp)
pred_map_before = get_prediction_map(model_fp, jpg_before, metadata_before, conus_fp, pixel_radius=4)
pred_map_after = get_prediction_map(model_fp, jpg_after, metadata_after, conus_fp, pixel_radius=4)

# IN THIS PHOTO: Black is forest (1), Grey is nonforest (2), White is water (3)
plt.imshow(pred_map_before, cmap = "gray")
plt.imshow(pred_map_after, cmap = "gray")


change_map = find_changes(pred_map_before, pred_map_after)
print(change_map[200, 300])
plt.imshow(change_map, cmap = "gray")
# Get JSON Array
def get_json_changes(change_map, metadata_fp):
    # Get bounds
    (ul_x, ul_y, lr_x, lr_y) = get_bounds(metadata_fp)
    width = len(change_map[0])
    height = len(change_map)

    inProj = Proj(init='epsg:5070')
    outProj = Proj(init='epsg:4269')
    json_arr = []

    # Get the lat/long of each region where there has been a change
    for y in range(0, height):
        for x in range(0, width):
            c = change_map[y, x]
            # Loss in Forest Cover
            if c == 1:
                # Get lat/long of the area
                curr_x = ul_x + ((x/width) * (lr_x-ul_x))
                curr_y = ul_y - ((y/height) * (ul_y-lr_y))
                (long, lat) = transform(inProj,outProj,curr_x,curr_y)
                json_change = {
        			"latitude": str(lat),
        			"longitude": str(long),
        			"color": "RED",
        			"weight": "5"
        		}
                json_arr.append(json_change)
                break
    return json_arr
x = get_json_changes(change_map, metadata_before)
