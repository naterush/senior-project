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
warnings.filterwarnings("ignore")

# Input: Takes in a JPG image with bounds given in an XML file, conus/non-conus map
# Produces a np array of training data for only the colored portions of the image
# Example instance produced in data: [averageR, averageG, averageB, conusLabel]
def get_labeled_data(satellite_filepath, metadata_filepath, conus_data_filepath, pixel_radius=5):

    if satellite_filepath.endswith(".jpg"):
        satellite_jpg_filepath = satellite_filepath
    elif satellite_filepath.endswith(".TIF"):
        # we have to actually convert the .TIF to a .jpg
        satellite_jpg_filepath = satellite_filepath.replace("TIF", "jpg")
        im = PIL.Image.open(satellite_filepath)
        rgbimg = PIL.Image.new("RGB", im.size)
        rgbimg.paste(im)
        rgbimg.save(satellite_jpg_filepath, 'JPEG')

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
            jpg_x = int(((x / max_x)*jpg_width))#, len(rgb_data[1] - 1))
            jpg_y = int(((y / max_y)*jpg_height))#, len(rgb_data) - 1)

            try:
                np.count_nonzero(rgb_data[jpg_y, jpg_x])
            except:
                print(f"Error {y}, {max_y}, {jpg_height}")
                print(rgb_data.shape)

            if np.count_nonzero(rgb_data[jpg_y, jpg_x]) != 0:
                s = rgb_data[jpg_y-pixel_radius:jpg_y+pixel_radius,
                                 jpg_x-pixel_radius:jpg_x+pixel_radius]

                avgR = np.average(s[:, :, 0])
                

                if np.isnan(avgR):
                    # Move onto next pixel if there are empty pixels in this radius
                    continue
                avgG = np.average(s[:, :, 1])
                avgB = np.average(s[:, :, 2])
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


def test_logreg_model(data, pct_train = 0.90):
    data_copy = data.copy()
    np.random.shuffle(data_copy)
    y_val_idx = len(data_copy[0]) - 1
    cutoff = int(len(data_copy)*pct_train)

    X_train = data_copy[:cutoff, :y_val_idx]
    Y_train = data_copy[:cutoff, y_val_idx]
    X_test = data_copy[cutoff:, :y_val_idx]
    Y_test = data_copy[cutoff:, y_val_idx]
    print("Training then testing Logistic Regression model on data")
    logreg = linear_model.LogisticRegression()
    logreg = logreg.fit(X_train, Y_train)

    Y_preds = logreg.predict(X_test)
    num_correct = len(Y_preds[(Y_test == Y_preds)])
    accuracy = (num_correct / len(Y_preds) )*100
    print("Logistic Regression accuracy on test data: " + str(round(accuracy, 2))+"%")

def test_decisiontree_model(data, pct_train = 0.90):
    data_copy = all_data.copy()
    np.random.shuffle(data_copy)
    y_val_idx = len(data_copy[0]) - 1
    cutoff = int(len(data_copy)*pct_train)

    X_train = data_copy[:cutoff, :y_val_idx]
    Y_train = data_copy[:cutoff, y_val_idx]
    X_test = data_copy[cutoff:, :y_val_idx]
    Y_test = data_copy[cutoff:, y_val_idx]
    print("Training then testing Decision Tree model on data")
    dt = tree.DecisionTreeClassifier().fit(X_train, Y_train)

    Y_preds = dt.predict(X_test)
    num_correct = len(Y_preds[(Y_test == Y_preds)])
    accuracy = (num_correct / len(Y_preds) )*100
    print("Decision Tree accuracy on test data: " + str(round(accuracy, 2))+"%")


forest_nonforest_img = 'conus_forest_nonforest.img'
#files = ['norcal1', 'norcal2', 'philly']
files = ['philly']
data = []
#for location in files:
r = get_labeled_data(
    'downloaded_sat_data/LT05_L1TP_120033_20100123_20161017_01_T1/LT05_L1TP_120033_20100123_20161017_01_T1_B1.TIF',
    'downloaded_sat_data/LT05_L1TP_120033_20100123_20161017_01_T1/LT05_L1TP_120033_20100123_20161017_01_T1_ANG.txt',
    forest_nonforest_img,
    pixel_radius=4
)
data.append(r)

data_tuple = tuple(data)
all_data = np.vstack(data_tuple)

test_logreg_model(all_data)
test_decisiontree_model(all_data)
