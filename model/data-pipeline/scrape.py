# This file demonstrates the capabilities of the landsat API
# URL: https://pypi.org/project/landsatxplore/

import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json
import os

username = 'ejperelmuter'
password = 'Simbacat3471'


class LandsatAPI(object):

    def __init__(self):
        self.landsat_api = landsatxplore.api.API(username, password)
        self.ee_api = EarthExplorer(username, password)

    def download(self, lat, long, output_folder="downloaded_sat_data"):
        scenes = self.landsat_api.search(
            dataset='LANDSAT_ETM_C1',
            latitude=lat,
            longitude=long,
            start_date='1995-01-01',
            end_date='2019-01-01',
            max_cloud_cover=10
        )
        entity_id = scenes[0]['entityId']
        print("Entity_id: " + str(entity_id))
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        self.ee_api.download(scene_id=entity_id, output_dir=output_folder)


api = LandsatAPI()
print(api.landsat_api)
d = api.download(34.885931, -79.804688)
