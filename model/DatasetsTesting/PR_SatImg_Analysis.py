from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp


def print_stats(ds):
    """
    Prints statistics about the given dataset
    """
    print("Dataset Name: " + ds.name)
    print("Dataset Mode: " + ds.mode)
    print("Band Count: " + str(ds.count))
    print("Dataset Width: " + str(ds.width))
    print("Dataset Height: " + str(ds.height))
    print("Dataset Bounds: ", ds.bounds)
    print("Dataset Transform: ", ds.transform)
    ul = ds.transform * (0, 0)
    print("Upper Left Corner: ", ul)
    lr = ds.transform * (ds.width, ds.height)
    print("Lower Right Corner: ", lr)
    {i: dtype for i, dtype in zip(ds.indexes, ds.dtypes)}

def img_to_df(img_name, max_lat, min_lat, max_long, min_long):
    """
    Given a path to a .img file, will return a pandas dataframe with (lat, long, biomass).
    max_lat, min_lat, max_long, min_long correspond to North, South, East, West respectively.
    """

    # read in the image
    dataset = rasterio.open(img_name)

    # band1 contains the biomass data we are interested in
    band1 = dataset.read(1)
    data = band1

    height = dataset.height
    width = dataset.width

    # longitude_delta is the length of each pixel in the x direction
    diff_long = max_long - min_long
    longitude_delta = diff_long / width

    # latitude_delta is the length of each pixel in the y direction
    diff_lat = max_lat - min_lat
    latitude_delta = diff_lat / height


    # loop over all the pixels in the map
    lat = max_lat
    long = min_long
    lat_long_data = []
    for x in range(0, width):
        lat = max_lat # Set longitude to far North (Top)
        for y in range(0, height):
            bm = data[y, x] # get the biomass at this lat, long
            if bm > 0:
                print(str(lat) + " " + str(long) + " " + str(bm))
            lat_long_data.append([lat, long, bm])
            lat = lat - latitude_delta
        long = long + longitude_delta

    # convert to a dataframe, and return
    return pd.DataFrame(data=lat_long_data, columns=['latitude', 'longitude', 'biomass'])

# north_latitude = float(18.3) + float(1442) / float(2925)
north_latitude = 18.792991452991455
# south_latitude = north_latitude - (float(3312) / float(2925))
south_latitude = 17.660683760683764
# west_longitude = -1*(float(66.5) + (float(2209)/float(2780)))
west_longitude = -67.29460431654677
# east_longitude = west_longitude + (float(4800)/float(2780))
east_longitude = -65.56798561151079
# x = east_longitude-west_longitude
# print(x)
# 192206 meters per 1.7266187050359747 degrees latitude
meters_per_pixel = 192206 / 3312 # ~50 meters, thus 5 pixels per pixel of the biomass image

# Finds the closest pixel to the to given lat, long
# Note: Only works in Northern Hemisphere, possibly just in our quadrant
# Returns the pixel as a tuple (x, y) from the top left corner (Northwest Corner)
def closest_pixel(lat, long, dimensions, borders):
    n_lat = borders[0]
    s_lat = borders[1]
    e_long = borders[2]
    w_long = borders[3]
    if long < w_long or long > e_long:
        # print("Error, longitude not in bounds, ", long)
        return (-1, -1)
    if lat < s_lat or lat > n_lat:
        # print("Error, latitude not in bounds, ", lat)
        return (-1, -1)
    width = dimensions[0]
    height = dimensions[1]

    x_diff = long - w_long
    y_diff = n_lat - lat

    x_pxls = (x_diff/(e_long - w_long)) * width
    y_pxls = (y_diff/(n_lat - s_lat)) * height
    return (int(x_pxls // 1), int(y_pxls // 1))

def average_rgb(image, x, y, width):
    shape = image.shape
    width = shape[1]
    height = shape[0]
    i = width // 2

    left_x = x - i
    if left_x < 0:
        left_x = 0

    right_x = x + i
    if right_x >= width:
        right_x = width - 1

    top_y = y - i
    if top_y < 0:
        top_y = 0

    bottom_y = y + i
    if bottom_y >= height:
        bottom_y = height - 1
    subpic = image[top_y:bottom_y, left_x:right_x]
    r_avg = np.average(subpic[:, :, 0])
    g_avg = np.average(subpic[:, :, 1])
    b_avg = np.average(subpic[:, :, 2])
    # r_avg = np.average(image[top_y:bottom_y, left_x:right_x, 0])
    # g_avg = np.average(image[top_y:bottom_y, left_x:right_x, 1])
    # b_avg = np.average(image[top_y:bottom_y, left_x:right_x, 2])
    # print('Average Red: ', r_avg)
    # print('Average Green: ', g_avg)
    # print('Average Blue: ', b_avg)
    return (int(r_avg), int(b_avg), int(g_avg))

borders = (north_latitude, south_latitude, east_longitude, west_longitude)
dims = (4800, 3312)
# t = closest_pixel(18, -66, dims, borders)
# print(t)

sat_image = image.imread('DatasetsTesting/data/PR_satellite_color.jpg')
# summarize shape of the pixel array
# print(sat_image.dtype)
# print(sat_image.shape)
# display the array of pixels as an image
# plt.imshow(data)
# plt.show()

bm_df = pd.read_csv('PR_Biomass_Coordinates_Dataset.csv')
# display(bm_df[:3])

# Only get the lat-long coords that are displayed in the color (JPG) satellite image
bounded_bm_df = bm_df[(bm_df['latitude'] < north_latitude) \
                & (bm_df['latitude'] > south_latitude) \
                & (bm_df['longitude'] > west_longitude) \
                & (bm_df['longitude'] < east_longitude)]
# display(bounded_bm_df[:3])

biomass_df = bounded_bm_df.reset_index()
r_vals = []
g_vals = []
b_vals = []
time = datetime.now()
l = []
n = len(biomass_df)
for i in range(0, n):
    # if i > 10:
    #     break
    if i % 100 == 0:
        # new_time = datetime.now()
        # d = new_time - time
        # print(str(d) + 'ms')
        # time = new_time
        print(str(i) + '/' + str(n))
    lat = biomass_df.loc[i, 'latitude']
    long = biomass_df.loc[i, 'longitude']
    bm = biomass_df.loc[i, 'biomass']
    # print('lat: ', lat)
    # print('long: ', long)
    # print('bm: ', bm)
    (x, y) = closest_pixel(lat, long, dims, borders)
    if x == -1:
        continue
    (r, g, b) = average_rgb(sat_image, x, y, 5)
    # print('Lat: ' + str(lat) + " Long: " + str(long) + " ==> ("+str(x)+', '+str(y)+") with avg RGB (" + str(r)+','+str(g)+','+str(b)+')')
    r_vals.append(r)
    g_vals.append(g)
    b_vals.append(b)
    l.append({'latitude':lat, 'longitude':long, 'biomass':bm, 'avg_red':r, 'avg_green':g, 'avg_blue':b})

biomass_avg_rgb_df = pd.DataFrame(data=l)
# biomass_df['average_red'] = r_vals
# biomass_df['average_green'] = g_vals
# biomass_df['average_blue'] = b_vals
# biomass_avg_rgb_df.to_csv(index_label=False)

display(biomass_avg_rgb_df[:10])

# average_rgb(data, 100, 100, 5)
