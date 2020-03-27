import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import pandas as pd
import rasterio
import tarfile
from PIL import Image
import numpy

zip_file = '/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/data-pipeline/downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1.tar.gz'

# tf = tarfile.open(zip_file)
# tf.next().name
# tf.extractall()


# tif_files = []
# text_files = []
# open_path = '/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/data-pipeline/downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1.tar.gz'
# extract_path = '/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/data-pipeline/extracted_data'
# with tarfile.open(open_path, "r:gz") as mytar:
#     for m in mytar.getnames():
#         print("File: " + str(m))
#         if m.endswith(".TIF"):
#             mytar.extract(m, path=extract_path)
#             tif_files.append(m)
#         if m.endswith(".txt"):
#             mytar.extract(m, path=extract_path)
#             text_files.append(m)

im = Image.open('/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/data-pipeline/extracted_data/LE07_L1TP_016036_19990719_20161003_01_T1_B1.TIF')
# im.show()
np_img = numpy.array(im)
print(np_img.shape)
print(np_img)
