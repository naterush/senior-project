import matplotlib.pyplot as plt
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