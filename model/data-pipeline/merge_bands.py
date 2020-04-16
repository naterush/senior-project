import PIL
import numpy as np
import matplotlib.pyplot as plt
import earthpy
import earthpy.spatial as es
import earthpy.plot as ep


all_landsat_post_bands = []
for i in range(1, 4):
    s = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/LC08_L1TP_014032_20200316_20200326_01_T1_B'
    s = s + str(i)+'.TIF'
    all_landsat_post_bands.append(s)
all_landsat_post_bands

(stack, meta) = es.stack(all_landsat_post_bands, 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/a.tif')
print(type(meta))

# all_landsat_post_bands.sort()
# Create an output array of all the landsat data stacked
landsat_post_fire_path = os.path.join("data", "cold-springs-fire",
                                      "outputs", "landsat_post_fire.tif")

# This will create a new stacked raster with all bands
land_stack, land_meta = es.stack(all_landsat_post_bands,
                                 landsat_post_fire_path)
