# This file demonstrates the capabilities of the landsat API
# URL: https://pypi.org/project/landsatxplore/

import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json
import os
import tarfile
import rasterio
import numpy as np
from affine import Affine
from pyproj import Proj, transform
from PIL import Image
import numpy as np
from pathlib import Path
import json
from create_labeled_data import get_labeled_data

lat = 39.952583
long = -75.165222

datasets = ['LANDSAT_TM_C1', 'LANDSAT_ETM_C1', 'LANDSAT_8_C1']


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
        band_2_path = self.tif_path_from_band(2)
        band_3_path = self.tif_path_from_band(3)
        band_4_path = self.tif_path_from_band(4)

        band_2_im = Image.open(band_2_path)
        band_2_arr = np.array(band_2_im)

        band_3_im = Image.open(band_3_path)
        band_3_arr = np.array(band_3_im)

        band_4_im = Image.open(band_4_path)
        band_4_arr = np.array(band_4_im)

        rgb_array = np.zeros((band_2_arr.shape[0], band_2_arr.shape[1], 3), 'uint8')
        rgb_array[..., 0] = band_4_arr
        rgb_array[..., 1] = band_3_arr
        rgb_array[..., 2] = band_2_arr

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

        scene_objs = []

        for scene_data in scenes[0:min(num_scenes, len(scenes))]:
            entity_id = scene_data['entityId']
            summary_id = scene_data["summary"].split(",")[0].split(":")[1][1:]
            
            # make the output directory if it doesn't exist
            if not output_folder.exists():
                output_folder.mkdir()

            # make an output folder for this specific scene
            if not (output_folder / summary_id).exists():
                (output_folder / summary_id).mkdir()

            self.ee_api.download(scene_id=entity_id, output_dir=output_folder / summary_id)
            scene_obj = Scene(output_folder / summary_id, scene_data)
            scene_objs.append(scene_obj)

        return scene_objs

"""
api = LandsatAPI()
d = api.download(38.8375, -120.8958, num_scenes=1, dataset="LANDSAT_ETM_C1")
api.logout()
for scene in d:
    scene.extract()
    labeled = scene.label()
    print(labeled)
"""