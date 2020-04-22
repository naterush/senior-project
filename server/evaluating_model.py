from sklearn import linear_model, tree
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import sklearn
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


def get_labeled_data(model_fp, satellite_jpg_filepath, metadata_xml_filepath, conus_data_filepath, pixel_radius=5):
    loaded_model = pickle.load(open(model_fp, 'rb'))
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
    prediction_map = forest_cover.copy()
    prediction_map[:, :] = 0
    max_x = len(forest_cover[0])
    max_y = len(forest_cover)
    results = np.zeros((max_x*max_y, 5))
    row_num = 0
    for x in range(0, max_x):
        if x % 100 == 0: print(str(x) + '/'+str(max_x))
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
                # results[row_num, 3] = 1 if (forest_label == 1) else 0
                results[row_num, 3] = forest_label
                results[row_num, 4] = loaded_model.predict([[avgR, avgG, avgB]])[0]
                prediction_map[y, x] = results[row_num, 4]
                row_num = row_num + 1

    all_results = results[:row_num].copy()
    return (all_results, prediction_map, forest_cover)


def get_labeled_data_quicker(model_fp, satellite_jpg_filepath, metadata_xml_filepath, conus_data_filepath, pixel_radius=5):
    loaded_model = pickle.load(open(model_fp, 'rb'))
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
    prediction_map = forest_cover.copy()
    prediction_map[:, :] = 0
    max_x = len(forest_cover[0])
    max_y = len(forest_cover)
    results = np.zeros((max_x*max_y, 2), dtype=int)
    row_num = 0
    print('On column ')
    for x in range(0, max_x):
        if x % 100 == 0: print(str(x) + '/'+str(max_x)+'...', end =" ")
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
                avgR = np.average(slice[:, :, 0])
                if np.isnan(avgR):
                    # Move onto next pixel if there are empty pixels in this radius
                    continue
                avgG = np.average(slice[:, :, 1])
                avgB = np.average(slice[:, :, 2])
                results[row_num, 0] = int(forest_label)
                results[row_num, 1] = int(loaded_model.predict([[avgR, avgG, avgB]])[0])
                prediction_map[y, x] = results[row_num, 1]
                row_num = row_num + 1

    all_results = results[:row_num].copy()
    return (all_results, prediction_map, forest_cover)


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

    print("Black percentage: " + str(((max_x*max_y) - row_num)/(max_x*max_y)))
    return prediction_map

def analyze_accuracy(model_fp, sat_img, meta_fp, conus_fp):
    (results, pred_map, real_map) = get_labeled_data(model_fp, sat_img, meta_fp, conus_fp)
    y_actual = int(results[:, 3])
    y_preds = int(results[:, 4])
    accuracy = sklearn.metrics.accuracy_score(y_actual, y_preds)
    grid = np.zeros((3, 3))
    # grid[A, B] = number of A instances predicted to be B
    for i in range(0, len(y_preds)):
        actual = y_actual[i]
        pred = y_preds[i]
        grid[actual, pred] = grid[actual, pred] + 1
    print('''                   **Predictions**
                Forest    Non-Forest  Water
    Real Forest  ['''+str(grid[0])+'''+
    Real No-Forest'''+str(grid[1])+'''
    Real Water    '''+str(grid[2])+''']
    ''')
    return accuracy

dtree = 'model/matching_coordinates/dtree_3class.sav'
logreg = 'model/matching_coordinates/logreg_model_3class.sav'
forest_nonforest_img = 'model/matching_coordinates/conus_forest_nonforest.img'
all_results_y = np.zeros((0, 2), dtype=int)
# files = ['cali1', 'cali2', 'cali3', 'philly', 'adam_cali']
files = ['eugene1', 'fresno1', 'lexington']

for f in files:
    # sat_img = 'model/matching_coordinates/sample_data/'+f+'.jpg'
    # meta_fp = 'model/matching_coordinates/sample_data/'+f+'.xml'
    sat_img = 'model/evaluation_data/'+f+'.jpg'
    meta_fp = 'model/evaluation_data/'+f+'.xml'
    t = time.time()
    (results, pred_map, real_map) = get_labeled_data_quicker(logreg, sat_img, meta_fp, forest_nonforest_img)
    d = time.time() - t

    y_actual = results[:, 0:1].copy()
    y_preds = results[:, 1:2].copy()
    acc = sklearn.metrics.accuracy_score(y_actual, y_preds)
    print(f"Image {f} -- {round(100*acc, 2)}% accuracy in {d} seconds")
    new_data = np.hstack((y_actual, y_preds))
    all_results_y = np.vstack((all_results_y, new_data))

# print(len(y_actual[y_actual == 1])/len(y_actual))
y_actual = all_results_y[:, 0]
y_preds = all_results_y[:, 1]
accuracy = sklearn.metrics.accuracy_score(y_actual, y_preds)
# print(f"Accuracy of decision tree 3 class: {round(accuracy*100, 2)}%")
print(f"Accuracy of logistic regression 3 class: {round(accuracy*100, 2)}%")
grid = np.zeros((3, 3))
# grid[A, B] = number of A instances predicted to be B
for i in range(0, len(y_preds)):
    actual = int(y_actual[i]) - 1
    pred = int(y_preds[i]) -1
    grid[actual, pred] = grid[actual, pred] + 1
print('''                   **Predictions**
            Forest    Non-Forest  Water
Real Forest  ['''+str(grid[0])+'''+
Real No-Forest'''+str(grid[1])+'''
Real Water    '''+str(grid[2])+''']
''')

# ******DECISION TREE*******
# Accuracy of decision tree 3 class: 74.89
# ON THESE FILES: ['cali1', 'cali2', 'cali3', 'philly', 'adam_cali']
#                    **Predictions**
#             Forest    Non-Forest  Water
# Real Forest  [[613501.  86436.   2709.]
# Real No-Forest[222182. 430071.  10113.]
# Real Water    [ 6474. 30277. 24992.]]

# Accuracy of decision tree 3 class: 78.53%, 75.96%, and 48.75% respectively
# ON THESE FILES: ['eugene1', 'fresno1', 'lexington']
#                    **Predictions**
#             Forest    Non-Forest  Water
# Real Forest  [[378193.  45906.   2048.]+
# Real No-Forest[264508. 273611.   7883.]
# Real Water    [3110. 2334. 1734.]]



# ******LOGISTIC REGRESSION*******
# 89.59% 80.32% 87.12% 33.51% 83.29% accuracy respectively
# ON THESE FILES: ['cali1', 'cali2', 'cali3', 'philly', 'adam_cali']
# Accuracy of logistic regression 3 class: 74.74%
#                    **Predictions**
#             Forest    Non-Forest  Water
# Real Forest  [[639361.  61295.   1990.]+
# Real No-Forest[266491. 391901.   3974.]
# Real Water    [16501. 10197. 35045.]]
#

# ACCURACIES: eugene: 79.96%, fresno1: 78.28%, lexington: 44.89%
# Accuracy of logistic regression 3 class: 66.55%
#                    **Predictions**
#             Forest    Non-Forest  Water
# Real Forest  [[395188.  29005.   1954.]+
# Real No-Forest[287401. 255241.   3360.]
# Real Water    [4796. 1108. 1274.]]
