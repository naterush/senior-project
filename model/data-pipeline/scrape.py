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
from pathlib import Path
import json

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
            dataset='LANDSAT_5',
            start_date='2018-01-01',
            end_date='2019-01-01',
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
            max_cloud_cover=1000
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



api = LandsatAPI()
d = api.download(38.8375, 120.8958, num_scenes=1)
api.logout()
for scene in d:
    scene.extract()