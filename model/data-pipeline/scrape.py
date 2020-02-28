# This file demonstrates the capabilities of the landsat API
# URL: https://pypi.org/project/landsatxplore/

import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json

username = 'ejperelmuter'
password = 'Simbacat3471'

class LandsatAPI(object):

    def __init__(self):
        print('yeet')
        self.landsat_api = landsatxplore.api.API(username, password)
        self.ee_api = EarthExplorer(username, password)
        print('yote')

    def download(self, lat, long):
        scenes = self.landsat_api.search(
            dataset='LANDSAT_ETM_C1',
            latitude=lat,
            longitude=long,
            start_date='1995-01-01',
            end_date='2019-01-01',
            max_cloud_cover=10
        )
        print(scenes)
        entity_id = scenes[0]['entityId']
        print("Entity_id: " + str(entity_id))
        self.ee_api.download(scene_id=entity_id, output_dir='/Users/ethanperelmuter/Desktop/senior-project(GitHub)/model/data-pipeline/downloaded_sat_data')


api = LandsatAPI()
print(api.landsat_api)
d = api.download(34.885931, -79.804688)
