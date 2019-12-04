from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp


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
time = datetime.now()

image = sat_image
shape = image.shape
print(shape)
width = shape[1]
height = shape[0]
# DEBUGGING Image from RGB vals
test_image = np.zeros((3312, 4800, 3), dtype=int)
subpic_width = 2

l = []
n = len(biomass_df)
for i in range(0, n):
    if i % 10000 == 0:
        print(str(i) + '/' + str(n))
    lat = biomass_df.loc[i, 'latitude']
    long = biomass_df.loc[i, 'longitude']
    bm = biomass_df.loc[i, 'biomass']

    n_lat = borders[0]
    s_lat = borders[1]
    e_long = borders[2]
    w_long = borders[3]
    if long < w_long or long > e_long:
        # print("Error, longitude not in bounds, ", long)
        continue
    if lat < s_lat or lat > n_lat:
        # print("Error, latitude not in bounds, ", lat)
        continue

    x_diff = long - w_long
    y_diff = n_lat - lat

    x_pxls = (x_diff/(e_long - w_long)) * width
    y_pxls = (y_diff/(n_lat - s_lat)) * height
    (x, y) = (int(x_pxls // 1), int(y_pxls // 1))
    # print('(x, y): '+str(x)+','+str(y))
    if x == -1 or y == -1:
        continue
    # (r, g, b) = average_rgb(sat_image, x, y, 5)



    left_x = x - subpic_width
    if left_x < 0:
        left_x = 0

    right_x = x + subpic_width
    if right_x >= width:
        right_x = width - 1

    top_y = y - subpic_width
    if top_y < 0:
        top_y = 0

    bottom_y = y + subpic_width
    if bottom_y >= height:
        bottom_y = height - 1
    # subpic = image[top_y:bottom_y, left_x:right_x]
    # r_avg = np.average(subpic[:, :, 0])
    # g_avg = np.average(subpic[:, :, 1])
    # b_avg = np.average(subpic[:, :, 2])
    r_avg = np.average(image[top_y:bottom_y, left_x:right_x, 0])
    g_avg = np.average(image[top_y:bottom_y, left_x:right_x, 1])
    b_avg = np.average(image[top_y:bottom_y, left_x:right_x, 2])

    (r, g, b) = (int(r_avg), int(g_avg), int(b_avg))
    test_image[y, x, 0] = r
    test_image[y, x, 1] = g
    test_image[y, x, 2] = b

    # print('Lat: ' + str(lat) + " Long: " + str(long) + " ==> ("+str(x)+', '+str(y)+") with avg RGB (" + str(r)+','+str(g)+','+str(b)+')')
    l.append({'latitude':lat, 'longitude':long, 'biomass':bm, 'avg_red':r, 'avg_green':g, 'avg_blue':b})

biomass_avg_rgb_df = pd.DataFrame(data=l)
# print('saving csv')
biomass_avg_rgb_df.to_csv('PuertoRico_Biomass_AvgRGB.csv', index_label=False)

plt.imshow(test_image)
plt.show()

# plt.imshow(image)
# plt.show()

# display(biomass_avg_rgb_df[biomass_avg_rgb_df['biomass'] != 0])

# average_rgb(data, 100, 100, 5)
