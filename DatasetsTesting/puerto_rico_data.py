import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp

# ****Uncomment this next section see the dataset ****
# df = pd.read_csv('PR_Biomass_Coordinates_Dataset.csv', index_col=False)
# df_biomass_exists = df[df['biomass'] > 0]
# display(df_biomass_exists)
# t1 = len(df)
# t2 = len(df_biomass_exists)
# pct = t2 / t1
# pct = (pct * 100)
# print(str(pct) + "% of the map contains biomass")



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


def img_to_df(img_name, 
        max_lat, 
        min_lat,
        max_long,
        min_long
    ):
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
            lat_long_data.append([lat, long, bm]) 
            lat = lat - latitude_delta
        long = long + longitude_delta

    # convert to a dataframe, and return
    return pd.DataFrame(data=lat_long_data, columns=['latitude', 'longitude', 'biomass'])


INPUT_IMG = "DatasetsTesting/Puerto_Rico_Biomass.img"
OUTPUT_CSV = 'PR_Biomass_Coordinates_Dataset.csv'

# The bounding box data can be gotten from the data download link, 
# searching for: North_Bounding_Coordinate, etc.

coord_biomass = img_to_df(
    INPUT_IMG,
    18.5542, # North
    17.7694, # South
    -65.13, # East Border
    -67.3228 # West Border,
)

coord_biomass.to_csv('PR_Biomass_Coordinates_Dataset.csv', index=False)