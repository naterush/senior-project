import PIL
import numpy as np
import matplotlib.pyplot as plt
plt.imshow(im2, cmap = "gray")


b2 = 'model/data-pipeline/downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1/LE07_L1TP_016036_19990719_20161003_01_T1_B2.TIF'
b3 = 'model/data-pipeline/downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1/LE07_L1TP_016036_19990719_20161003_01_T1_B3.TIF'
b4 = 'model/data-pipeline/downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1/LE07_L1TP_016036_19990719_20161003_01_T1_B4.TIF'
# satellite_jpg_filepath = satellite_filepath.replace("TIF", "jpg")
im2 = PIL.Image.open(b2)
im3 = PIL.Image.open(b3)
im4 = PIL.Image.open(b4)
b_arr = np.array(im2)
g_arr = np.array(im3)
r_arr = np.array(im4)
shape = b_arr.shape
print(shape)
rgb = np.zeros((shape[0], shape[1], 3))
rgb[:, :, 0] = r_arr
rgb[:, :, 1] = g_arr
rgb[:, :, 2] = b_arr
print(rgb[])

print(type(im2))

plt.imshow(rgb)

rgbArray = np.zeros((512,512,3), 'uint8')


rgbArray[..., 0] = r*256
rgbArray[..., 1] = g*256
rgbArray[..., 2] = b*256
img = Image.fromarray(rgbArray)
img.save('myimg.jpeg')
