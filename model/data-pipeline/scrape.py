# This file demonstrates the capabilities of the landsat API
# URL: https://pypi.org/project/landsatxplore/

import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json
import os

username = 'ejperelmuter'
password = 'Sapling#2020'

lat = 39.952583
long = -75.165222

datasets = ['LANDSAT_TM_C1', 'LANDSAT_ETM_C1', 'LANDSAT_8_C1']


class LandsatAPI(object):

    def __init__(self):
        self.landsat_api = landsatxplore.api.API(username, password)
        self.ee_api = EarthExplorer(username, password)

    def download(self, lat, long, output_folder="downloaded_sat_data"):
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
        print("Entity_id: " + str(entity_id))

        # if not os.path.exists(output_folder):
        #     os.mkdir(output_folder)
        #
        # self.ee_api.download(scene_id=entity_id, output_dir=output_folder)


api = LandsatAPI()
d = api.download(lat, long)
