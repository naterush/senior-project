import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import datetime

# ****Uncomment this next section see the dataset ****
# df = pd.read_csv('PR_Biomass_Coordinates_Dataset.csv', index_col=False)
# df_biomass_exists = df[df['biomass'] > 0]
# display(df_biomass_exists)
# t1 = len(df)
# t2 = len(df_biomass_exists)
# pct = t2 / t1
# pct = (pct * 100)
# print(str(pct) + "% of the map contains biomass")



img_name = "DatasetsTesting/Puerto_Rico_Biomass.img"
output_filename = 'image1.png'

def print_stats(ds):
    print("Dataset Name: " + ds.name)
    print("Dataset Mode: " + ds.mode)
    print("Band Count: " + str(ds.count))
    print("Dataset Width: " + str(ds.width))
    print("Dataset Height: " + str(ds.height))
    print("Dataset Bounds: ", dataset.bounds)
    print("Dataset Transform: ", dataset.transform)
    ul = dataset.transform * (0, 0)
    print("Upper Left Corner: ", ul)
    lr = dataset.transform * (dataset.width, dataset.height)
    print("Lower Right Corner: ", lr)
    {i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)}

dataset = rasterio.open(img_name)
height = dataset.height
width = dataset.width
# print_stats(dataset)

band1 = dataset.read(1)
# Display.
plt.imshow(band1, cmap = "gray")
plt.savefig(output_filename)
plt.show()

# Goal: Build table of (latitude, longitude, pixel value) tuples

data = band1
max_lat = 18.5542 # North
min_lat = 17.7694 # South

max_long = -65.13 # East Border
min_long = -67.3228 # West Border
diff_x = max_long - min_long
print(diff_x/width)
longitude_delta = diff_x/929

diff_y = max_lat - min_lat
print(diff_y/height)
latitude_delta = diff_y/349
lat = 18.5542
long = -67.3228
lat_long_data = []
time = datetime.datetime.now()
for x in range(0, 929):
    lat = 18.5542 # Set longitude to far North (Top)
    for y in range(0, 349):
        bm = data[y, x]
        # coord_biomass = coord_biomass.append({'latitude': lat, 'longitude': long, 'biomass':bm}, ignore_index=True)
        lat_long_data.append([lat, long, bm])
        lat = lat - latitude_delta
    long = long + longitude_delta
time_diff = datetime.datetime.now() - time
print('Time to process data: ', time_diff.total_seconds())
coord_biomass = pd.DataFrame(data=lat_long_data, columns=['latitude', 'longitude', 'biomass'])
# coord_biomass.to_csv('PR_Biomass_Coordinates_Dataset.csv', index=False)
display(coord_biomass[coord_biomass['biomass'] > 0])
# display(coord_biomass[coord_biomass['biomass'] > 0])
