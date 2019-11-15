import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import rasterio.warp

img_name = "Puerto_Rico_Biomass.img"
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
print_stats(dataset)

band1 = dataset.read(1)
shape = (349, 929)
print(band1.shape)
image = band1.reshape(shape)

# Display.
plt.imshow(image, cmap = "gray")
plt.savefig(output_filename)
plt.show()






# Read the dataset's valid data mask as a ndarray.
mask = dataset.dataset_mask()

# Extract feature shapes and values from the array.
for geom, val in rasterio.features.shapes(
        mask, transform=dataset.transform):

    # Transform shapes from the dataset's own coordinate
    # reference system to CRS84 (EPSG:4326).
    geom = rasterio.warp.transform_geom(
        dataset.crs, 'EPSG:4326', geom, precision=6)

    # Print GeoJSON shapes to stdout.
    print(geom)

fp = "Puerto_Rico_Biomass.img"
raster = rasterio.open(fp)

type(raster)






for i in range(1, 502960):
    if 502960 % i == 0:
        print("Coordinates: " + str(502960/i) + " by " + str(i))

# Parameters.
input_filename = "Puerto_Rico_Biomass.img"
shape = (6287, 80) # matrix size
dtype = np.dtype(np.int8) # big-endian unsigned integer (16bit)
output_filename = "TestImage.PNG"

# Reading.
fid = open(input_filename, 'rb')
data = np.fromfile(fid, dtype)
image = data.reshape(shape)
# image = data

# Display.
plt.imshow(image, cmap = "gray")
plt.savefig(output_filename)
plt.show()
