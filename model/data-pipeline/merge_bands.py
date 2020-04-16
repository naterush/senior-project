import PIL
import numpy as np
import matplotlib.pyplot as plt
import earthpy
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio


all_landsat_post_bands = []
for i in range(1, 5):
    s = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/LC08_L1TP_014032_20200316_20200326_01_T1_B'
    s = s + str(i)+'.TIF'
    all_landsat_post_bands.append(s)
all_landsat_post_bands
output_fp = 'model/data-pipeline/downloaded_sat_data/ethan_test_LC08/a.tif'
(stack, meta) = es.stack(all_landsat_post_bands, output_fp)

with rio.open(output_fp) as src:
    landsat_post_fire = src.read()
    ep.plot_rgb(landsat_post_fire,
            rgb=[3, 2, 1],
            title="RGB Composite Image\n Post Fire Landsat Data",
            stretch = True)
    plt.show()
