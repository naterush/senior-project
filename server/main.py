import sys
import json
import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json
import os
import tarfile
import rasterio
import numpy as np
from pyproj import Proj, transform
from PIL import Image
import PIL
import numpy as np
from pathlib import Path
import json
from pyproj import Proj, transform
import xmltodict
import pickle
import warnings
warnings.filterwarnings("ignore")

model_filepath = 'dtree_3class.sav'
conus_filepath = 'conus_forest_nonforest.img'


_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)


def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image



def norm(band):
    band_min, band_max = band.min(), band.max()
    return ((band - band_min)/(band_max - band_min))


# from create_labeled_data import get_labeled_data
class LandsatAPI(object):
    def __init__(self, username="ejperelmuter", password="Sapling#2020"):
        self.landsat_api = landsatxplore.api.API(username, password)
        self.ee_api = EarthExplorer(username, password)

    def logout(self):
        self.landsat_api.logout()
        self.ee_api.logout()

    def download(
            self,
            lat,
            long,
            output_folder="downloaded_sat_data",
            dataset='LANDSAT_8_C1',
            start_date='2016-01-01',
            end_date='2018-01-01',
            num_scenes=1 # the number of scenes to grab, starting from the first
        ):
        # convert the output_folder to a path, for easier handling
        output_folder = Path(output_folder)

        scenes = self.landsat_api.search(
            dataset=dataset,
            latitude=lat,
            longitude=long,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=10
        )
        print("Number scenes found: " + str(len(scenes)))

        # make the output directory if it doesn't exist
        if not output_folder.exists():
            output_folder.mkdir()

        scene_objs = []

        for scene_data in scenes[0:min(num_scenes, len(scenes))]:
            entity_id = scene_data['entityId']
            summary_id = scene_data["summary"].split(",")[0].split(":")[1][1:]

            # make an output folder for this specific scene
            if not (output_folder / summary_id).exists():
                (output_folder / summary_id).mkdir()

                self.ee_api.download(scene_id=entity_id, output_dir=output_folder / summary_id)
                scene_obj = Scene(output_folder / summary_id, scene_data)
                scene_obj.extract()
                scene_objs.append(scene_obj)
            else:
                scene_obj = Scene(output_folder / summary_id)
                scene_objs.append(scene_obj)

        return scene_objs

class Scene():

    def __init__(self, folder_path, scene_data=None):
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)

        self.folder_path = folder_path

    def extract(self):
        if len(list(self.folder_path.iterdir())) > 2:
            # we don't need to do anything if this was already extracted
            print("Already extracted! Returning.")
            return

        # We need to extract the text file that ends in ANG.txt
        # as well as the bands 2, 3, 4,
        def to_download(tarinfo):
            name = tarinfo.name
            if name.endswith("ANG.txt"):
                return True
            if name.endswith("B2.TIF"):
                    return True
            if name.endswith("B3.TIF"):
                    return True
            if name.endswith("B4.TIF"):
                    return True
            return False

        # extract the tar file
        tar_file = list(x for x in self.folder_path.iterdir() if x.suffix == ".gz")[0]
        with tarfile.open(tar_file, "r:gz") as mytar:
            print([m for m in mytar.getmembers() if to_download(m)])
            mytar.extractall(path=self.folder_path, members=[m for m in mytar.getmembers() if to_download(m)])

        print(f"Extracted in {self.folder_path}")

        # delete the tar file
        os.remove(tar_file)

    def tif_path_from_band(self, band_num):
        """
        Returns a path to the TIF file that with that band, or
        None of that band_num does not exist
        """

        for path in self.folder_path.iterdir():
            if path.suffix == ".TIF":
                if path.name.endswith(f"B{band_num}.TIF"):
                    return Path(path)

        return None

    def metadata_path_str(self):
        for path in self.folder_path.iterdir():
            if path.suffix == ".txt":
                if path.name.endswith("ANG.txt"):
                    return str(path)
        return None

    def get_rgb_array(self):
        band_r_path = self.tif_path_from_band(4)
        band_g_path = self.tif_path_from_band(3)
        band_b_path = self.tif_path_from_band(2)

        band_r_im = Image.open(band_r_path)
        band_r_arr = np.array(band_r_im)

        band_g_im = Image.open(band_g_path)
        band_g_arr = np.array(band_g_im)

        band_b_im = Image.open(band_b_path)
        band_b_arr = np.array(band_b_im)

        rgb_array = np.zeros((band_r_arr.shape[0], band_r_arr.shape[1], 3), 'uint8')
        rgb_array[..., 0] = band_r_arr
        rgb_array[..., 1] = band_g_arr
        rgb_array[..., 2] = band_b_arr

        return rgb_array

    def write_img(self, img_path="myimg.jpeg"):
        band_2_path = self.tif_path_from_band(2)
        band_3_path = self.tif_path_from_band(3)
        band_4_path = self.tif_path_from_band(4)

        band_2_im = Image.open(band_2_path)
        band_2_arr = norm(np.array(band_2_im).astype(np.float))

        band_3_im = Image.open(band_3_path)
        band_3_arr = norm(np.array(band_3_im).astype(np.float))

        band_4_im = Image.open(band_4_path)
        band_4_arr = norm(np.array(band_4_im).astype(np.float))

        rgb = np.dstack((band_4_arr,band_3_arr,band_2_arr))
        del band_2_arr, band_3_arr, band_4_arr

        img = toimage(rgb, cmin=np.percentile(rgb,2),
               cmax=np.percentile(rgb,98), mode="RGB")

        img.save(img_path)

    def label(self):
        rgb_array = self.get_rgb_array()
        metadata_filepath = self.metadata_path_str()
        conus_data_filepath = "conus_forest_nonforest.img"
        labeled_data = get_labeled_data(rgb_array, metadata_filepath, conus_data_filepath, pixel_radius=5)
        return labeled_data

def get_prediction_map(model_filepath, satellite_jpg_filepath, metadata_filepath, conus_data_filepath, pixel_radius=4):
    loaded_model = pickle.load(open(model_filepath, 'rb'))

    # Get the bounding coordinates from the metadata file
    if metadata_filepath.endswith(".xml"):
        with open(metadata_filepath) as fd:
            metadata = xmltodict.parse(fd.read())

        # Get the Albers Equal Area bounds of the satellite image
        projection_bounds = metadata['ard_metadata']['tile_metadata']['global_metadata']['projection_information']
        ul_x = float(projection_bounds['corner_point'][0]['@x'])
        ul_y = float(projection_bounds['corner_point'][0]['@y'])
        lr_x = float(projection_bounds['corner_point'][1]['@x'])
        lr_y = float(projection_bounds['corner_point'][1]['@y'])
    elif metadata_filepath.endswith(".txt"):
        with open(metadata_filepath) as fd:
            lines = [l.strip() for l in fd.readlines()]

        ul_line = [l for l in lines if l.startswith("UL_CORNER")][0].split("=")[1].strip()[1:-1]
        lr_line = [l for l in lines if l.startswith("LR_CORNER")][0].split("=")[1].strip()[1:-1]
        ul_x = float(ul_line.split(",")[0])
        ul_y = float(ul_line.split(",")[1])
        lr_x = float(lr_line.split(",")[0])
        lr_y = float(lr_line.split(",")[1])

    # Open the satellite image
    img = PIL.Image.open(satellite_jpg_filepath)
    rgb_data = np.asarray(img)
    jpg_width = rgb_data.shape[1]
    jpg_height = rgb_data.shape[0]

    # Open the Conus/Non-Conus Dataset
    ds = rasterio.open(conus_data_filepath)
    band1 = ds.read(1)

    # For each labeled point in the Conus dataset within the JPG image, create entry
    start_x = int((ul_x - ds.bounds.left)//250)
    end_x = int(start_x + ((lr_x - ul_x)//250))
    start_y = int((ds.bounds.top - ul_y)//250)
    end_y = int(start_y + ((ul_y - lr_y)//250))

    forest_cover = band1[start_y:end_y, start_x:end_x].copy()
    prediction_map = forest_cover.copy()
    prediction_map[:, :] = 0

    max_x = len(forest_cover[0])
    max_y = len(forest_cover)
    row_num = 0
    for x in range(0, max_x):
        if x % 100 == 0: print('On column ' + str(x) + '/' + str(max_x) + ' of image')
        for y in range(0, max_y):
            # Get the appropriate coordinates within the JPG image
            jpg_x = int(((x / max_x)*jpg_width))
            jpg_y = int(((y / max_y)*jpg_height))
            pixel = rgb_data[jpg_y, jpg_x]
            if pixel[0] != 0 and pixel[1] != 0 and pixel[2] != 0:
                pixel_square = rgb_data[jpg_y-pixel_radius:jpg_y+pixel_radius,
                                 jpg_x-pixel_radius:jpg_x+pixel_radius]

                avgR = np.average(pixel_square[:, :, 0])
                if np.isnan(avgR):
                    # Move onto next pixel if there are empty pixels in this radius
                    continue
                avgG = np.average(pixel_square[:, :, 1])
                avgB = np.average(pixel_square[:, :, 2])
                # albers_x = ds.bounds.left + (250*(start_x+x))
                # albers_y = ds.bounds.top - (250*(start_y+y))
                pred = loaded_model.predict([[avgR, avgG, avgB]])
                prediction_map[y, x] = pred[0]
                row_num = row_num + 1 # Track how full the photo is

    print("Black percentage: " + str(((max_x*max_y) - row_num)/(max_x*max_y)))
    return prediction_map

def find_changes(region1, region2, grid_size=60, thresh_pct=0):
    change_map = np.zeros(region1.shape, dtype=int)
    height = region1.shape[0]
    width = region1.shape[1]
    pixels_per_square = (height//grid_size) * (width//grid_size)
    print(pixels_per_square)
    thresh = pixels_per_square * 0.50
    for y_top_left in range(0, height, height//grid_size):
        for x_top_left in range(0, width, width//grid_size):
            r1_square = region1[y_top_left:y_top_left+grid_size,
                                        x_top_left:x_top_left+grid_size]
            r2_square = region2[y_top_left:y_top_left+grid_size,
                                        x_top_left:x_top_left+grid_size]
            change_map_region = change_map[y_top_left:y_top_left+grid_size,
                                        x_top_left:x_top_left+grid_size]
            # Get only the pixels predictions changed
            s1 = np.sum(r1_square)
            s2 = np.sum(r2_square)
            if (np.count_nonzero(r1_square) <= thresh) or (np.count_nonzero(r2_square) <= thresh):
                change_map_region[:, :] = 0
                continue
            # Ignore if either square is mostly black
            if s1 <= pixels_per_square or s2 <= pixels_per_square:
                change_map_region[:, :] = 0
            # TODO: Add some threshold of change required to flag it
            elif s1 < s2 and (s2-s1 > thresh):
                # There has been a LOSS in forest cover
                change_map_region[:, :] = -1
            elif s2 < s1 and (s1 - s2 > thresh):
                # There has been an GAIN in forest cover
                change_map_region[:, :] = 1
            else:
                # NO CHANGE
                change_map_region[:, :] = 0
    return change_map

def get_json_changes(change_map, metadata_fp, grid_size=60):
    # Get bounds
    (ul_x, ul_y, lr_x, lr_y) = get_bounds(metadata_fp)
    width = len(change_map[0])
    height = len(change_map)

    inProj = Proj(init='epsg:5070')
    outProj = Proj(init='epsg:4269')
    json_arr = []

    # Get the lat/long of each region where there has been a change
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            change_map_square = change_map[y:y + grid_size, x:x + grid_size]
            change_sum = np.sum(change_map_square)
            print(y, x)

            curr_x = ul_x + ((x/width) * (lr_x-ul_x))
            curr_y = ul_y - ((y/height) * (ul_y-lr_y))
            (long, lat) = transform(inProj,outProj,curr_x,curr_y)

            if change_sum != 0:
                if change_sum <= 0:
                    json_change = {
                        "latitude": str(lat),
                        "longitude": str(long),
                        "color": "RED",
                        "weight": "5"
                    }
                if change_sum >= 0:
                    json_change = {
                        "latitude": str(lat),
                        "longitude": str(long),
                        "color": "GREEN",
                        "weight": "5"
                    }

                json_arr.append(json_change)
    return [change for change in json_arr if change is not None]

def get_bounds(metadata_filepath):
    # Get the bounding coordinates from the metadata file
    if metadata_filepath.endswith(".xml"):
        with open(metadata_filepath) as fd:
            metadata = xmltodict.parse(fd.read())

        # Get the Albers Equal Area bounds of the satellite image
        projection_bounds = metadata['ard_metadata']['tile_metadata']['global_metadata']['projection_information']
        ul_x = float(projection_bounds['corner_point'][0]['@x'])
        ul_y = float(projection_bounds['corner_point'][0]['@y'])
        lr_x = float(projection_bounds['corner_point'][1]['@x'])
        lr_y = float(projection_bounds['corner_point'][1]['@y'])
    elif metadata_filepath.endswith(".txt"):
        with open(metadata_filepath) as fd:
            lines = [l.strip() for l in fd.readlines()]

        ul_line = [l for l in lines if l.startswith("UL_CORNER")][0].split("=")[1].strip()[1:-1]
        lr_line = [l for l in lines if l.startswith("LR_CORNER")][0].split("=")[1].strip()[1:-1]
        ul_x = float(ul_line.split(",")[0])
        ul_y = float(ul_line.split(",")[1])
        lr_x = float(lr_line.split(",")[0])
        lr_y = float(lr_line.split(",")[1])
        # TODO: ZONE HERE
        inProj = Proj(proj="utm",zone=18,ellps="WGS84", south=False)
        outProj = Proj(init='epsg:5070')

        print("(ul_x, ul_y): ", (ul_x, ul_y))
        (ul_x, ul_y) = transform(inProj,outProj,ul_x,ul_y)
        print("after (ul_x, ul_y): ", (ul_x, ul_y))

        print("(lr_x, lr_y): ", (lr_x, lr_y))
        (lr_x, lr_y) = transform(inProj,outProj,lr_x,lr_y)
        print("after (lr_x, lr_y): ", (lr_x, lr_y))

    return (ul_x, ul_y, lr_x, lr_y)

def main():
    if (len(sys.argv)) != 3:
        print("Usage: python3 main.py <lat> <long>")
        return

    # Get the user given coordinates
    lat = float(sys.argv[1])
    lng = float(sys.argv[2])

    print(f"Getting images at coordinates ({lat}, {lng})")

    # Download 2 images at these coordinates
    api = LandsatAPI()
    print("Downloading the 2016 scene...")
    old_scene = api.download(
        lat,
        lng,
        num_scenes=1,
        start_date="2016-06-01",
        end_date="2016-09-01"
    )[0]

    # WRS Row Path is the location of the image, which we use as the key to cache computations
    WRS_ROW_PATH = str(old_scene.folder_path).split("/")[1].split("_")[2]

    # we check the cache to see if we've already done the computation
    if os.path.exists(f"output/{WRS_ROW_PATH}.txt"):
        print(f"output/{WRS_ROW_PATH}.txt")
        exit(0)

    old_scene_metadata = old_scene.metadata_path_str()
    before_jpg_filepath = str(old_scene.folder_path) + '/before_img.jpg'
    old_scene.write_img(before_jpg_filepath)

    print("Downloading the 2019 scene...")
    new_scene = api.download(
        lat,
        lng,
        num_scenes=1,
        start_date="2019-06-01",
        end_date="2019-09-01"
    )[0]
    new_scene_metadata = new_scene.metadata_path_str()
    after_jpg_filepath = str(new_scene.folder_path) + '/after_img.jpg'
    new_scene.write_img(after_jpg_filepath)

    # Run predictions on both downloaded images
    prediction_map_before = get_prediction_map(model_filepath, before_jpg_filepath, old_scene_metadata, conus_filepath)
    prediction_map_after = get_prediction_map(model_filepath, after_jpg_filepath, new_scene_metadata, conus_filepath)

    print("getting the change map")
    change_map = find_changes(prediction_map_before, prediction_map_after)
    print("getting the json changes")
    json_changes = get_json_changes(change_map, old_scene_metadata)

    data_real = json.dumps(json_changes)

    data_fake = json.dumps([
        {
            "latitude": "40.416775",
            "longitude": "-3.70379",
            "color": "GREEN",
            "weight": "6"
        },
        {
            "latitude": "41.385064",
            "longitude": "2.173403",
            "color": "GREEN",
            "weight": "2"
        },
        {
            "latitude": "52.130661",
            "longitude": "-3.783712",
            "color": "GREEN",
            "weight": "2"
        },
        {
            "latitude": "55.378051",
            "longitude": "-3.435973",
            "color": "GREEN",
            "weight": "8"
        },
        {
            "latitude": "-40.900557",
            "longitude": "-174.885971",
            "color": "GREEN",
            "weight": "6"
        },
        {
            "latitude": "40.714353",
            "longitude": "-74.005973",
            "color": "RED",
            "weight": "6"
        }
    ])

    if not os.path.exists("output"):
        os.mkdir("output")

    with open(f"output/{WRS_ROW_PATH}.txt", "w+") as f:
        f.write(data_real)

    print(f"output/{WRS_ROW_PATH}.txt")
    #NOTE: Do not print below this line. The server relies on this being the last print


if __name__ == "__main__":
    main()
