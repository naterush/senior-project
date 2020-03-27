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

username = 'ejperelmuter'
password = 'Simbacat3471'

lat = 39.952583
long = -75.165222

datasets = ['LANDSAT_TM_C1', 'LANDSAT_ETM_C1', 'LANDSAT_8_C1']


class Scene():

    def __init__(self, folder_path):
        self.folder_path = folder_path

    def extract(self):
        if len(list(self.folder_path.iterdir())) > 1:
            # we don't need to do anything if this was already extracted
            print("Already extracted! Returning.")
            return

        # extract the tar file
        tar_file = list(self.folder_path.iterdir())[0]
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
    
    def test(self):
        band_1 = self.tif_path_from_band(1)

        # Read raster
        with rasterio.open(band_1) as r:
            T0 = r.transform  # upper-left pixel corner affine transform
            p1 = Proj(r.crs)
            A = r.read()  # pixel values
            print(f"T0: {T0}")
            print(f"p1: {p1}")
            print(f"A: {A}")

        return

        # All rows and columns
        cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

        # Get affine transform for pixel centres
        T1 = T0 * Affine.translation(0.5, 0.5)
        # Function to convert pixel row/column index (from 0) to easting/northing at centre
        rc2en = lambda r, c: (c, r) * T1

        # All eastings and northings (there is probably a faster way to do this)
        eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)

        # Project all longitudes, latitudes
        p2 = Proj(proj='latlong' ,datum='WGS84')
        longs, lats = transform(p1, p2, eastings, northings)
        longs.tofile("longs.txt")
        lats.tofile("lats.txt")


def lat_long():
    longs = np.fromfile("longs.txt")
    lats = np.fromfile("lats.txt")
    return longs, lats


class LandsatAPI(object):

    def __init__(self):
        self.landsat_api = landsatxplore.api.API(username, password)
        self.ee_api = EarthExplorer(username, password)

    def download(self, lat, long, output_folder="downloaded_sat_data"):
        # convert the output_folder to a path, for easier handling
        output_folder = Path(output_folder)

        scenes = self.landsat_api.search(
            dataset=datasets[2],
            latitude=lat,
            longitude=long,
            start_date='2018-01-01',
            end_date='2019-01-01',
            max_cloud_cover=1000
        )
        print(scenes)
        print("Number found scenes" + str(len(scenes)))

        for scene in scenes:
            print(scene['acquisitionDate'])

        entity_id = scenes[0]['entityId']
        summary_id = scenes[0]["summary"].split(",")[0].split(":")[1][1:]
        
        # make the output directory if it doesn't exist
        if not output_folder.exists():
            output_folder.mkdir()

        # make an output folder for this specific scene
        if not (output_folder / summary_id).exists():
            (output_folder / summary_id).mkdir()

        self.ee_api.download(scene_id=entity_id, output_dir=output_folder / summary_id)
        return Scene(output_folder / summary_id)


api = LandsatAPI()
d = api.download(34.885931, -79.804688)
d.extract()
#scene = Scene(Path("./downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1"))
#scene.test()
exit(1)

"""
OLD CODE, NOT SURE IF WE NEED!

with tarfile.open('./downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1.tar.gz', "r:gz") as mytar:
    mytar.extractall(path="downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1")

for file_name in os.listdir("downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1"):
    if file_name.endswith(".TIF"):
        Image.MAX_IMAGE_PIXELS = 220835761
        im = Image.open("downloaded_sat_data/LE07_L1TP_016036_19990719_20161003_01_T1/" + file_name)
        im.show()

api = LandsatAPI()
d = api.download(lat, long)

"""
