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
import numpy as np
from pathlib import Path
import json
from pyproj import Proj, transform
import xmltodict
import pickle
import warnings
warnings.filterwarnings("ignore")

model_filepath = 'model/matching_coordinates/dtree_3class.sav'
conus_filepath = 'server/conus_forest_nonforest.img'

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
        print("Number found scenes" + str(len(scenes)))

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

        if scene_data is None:
            # if this object is created directly from the folder path,
            # it must already be extracted, so we read scene data
            with open(self.folder_path / "scene_data.txt", "r") as f:
                self.scene_data = json.loads(f.read())
        else:
            self.scene_data = scene_data

            # then we write the scene data
            with open(self.folder_path / "scene_data.txt", "w+") as f:
                f.write(json.dumps(self.scene_data))

    def extract(self):
        if len(list(self.folder_path.iterdir())) > 2:
            # we don't need to do anything if this was already extracted
            print("Already extracted! Returning.")
            return

        # extract the tar file
        tar_file = list(x for x in self.folder_path.iterdir() if x.suffix == ".gz")[0]
        with tarfile.open(tar_file, "r:gz") as mytar:
            mytar.extractall(path=self.folder_path)

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
        img = Image.fromarray(self.get_rgb_array())
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

def find_changes(region1, region2, grid_size=60, thresh_pct = 0):
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
                change_map_region[:, :] = 1
            elif s2 < s1 and (s1 - s2 > thresh):
                # There has been an GAIN in forest cover
                change_map_region[:, :] = 2
            else:
                # NO CHANGE
                change_map_region[:, :] = 0
    return change_map

def get_json_changes(change_map, metadata_fp):
    # Get bounds
    (ul_x, ul_y, lr_x, lr_y) = get_bounds(metadata_fp)
    width = len(change_map[0])
    height = len(change_map)

    inProj = Proj(init='epsg:5070')
    outProj = Proj(init='epsg:4269')
    json_arr = []

    # Get the lat/long of each region where there has been a change
    for y in range(0, height):
        for x in range(0, width):
            c = change_map[y, x]
            # Loss in Forest Cover
            if c == 1:
                # Get lat/long of the area
                curr_x = ul_x + ((x/width) * (lr_x-ul_x))
                curr_y = ul_y - ((y/height) * (ul_y-lr_y))
                (long, lat) = transform(inProj,outProj,curr_x,curr_y)
                json_change = {
        			"latitude": str(lat),
        			"longitude": str(long),
        			"color": "RED",
        			"weight": "5"
        		}
                json_arr.append(json_change)
    return json_arr

def main():
    if (len(sys.argv)) != 3:
        print("Usage: python3 main.py <lat> <long>")
        return

    # Get the user given coordinates
    lat = float(sys.argv[1])
    lng = float(sys.argv[2])

    # Download 2 images at these coordinates
    api = LandsatAPI()
    print("Downloading the 2016 scene...")
    old_scene = api.download(
        lat,
        lng,
        num_scenes=1,
        start_date="2016-4-1",
        end_date="2017-1-1"
    )[0]
    old_scene_metadata = old_scene.metadata_path_str()
    before_jpg_filepath = 'before_img.jpg'
    old_scene.write_img(before_jpg_filepath)

    print("Downloading the 2019 scene...")
    new_scene = api.download(
        lat,
        lng,
        num_scenes=1,
        start_date="2020-1-1",
        end_date="2020-4-17"
    )[0]
    new_scene_metadata = new_scene.metadata_path_str()
    after_jpg_filepath = 'after_img.jpg'
    new_scene.write_img(after_jpg_filepath)

    # Run predictions on both downloaded images
    prediction_map_before = get_prediction_map(model_filepath, before_jpg_filepath, old_scene_metadata, conus_filepath)
    prediction_map_after = get_prediction_map(model_filepath, after_jpg_filepath, new_scene_metadata, conus_filepath)

    change_map = find_changes(prediction_map_before, prediction_map_after)
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

    print(data_real)


if __name__ == "__main__":
    main()
