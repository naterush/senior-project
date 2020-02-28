# This file demonstrates the capabilities of the landsat API
# URL: https://pypi.org/project/landsatxplore/

import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json

username = 'ejperelmuter'
password = 'Simbacat3471'

class API(object):

    def __init__(self):
        self.landsat_api = landsatxplore.api.API(username, password)
        self.ee_api = EarthExplorer(username, password)

    def download(self, lat, long):
        scenes = self.landsat_api.search(
            dataset='LANDSAT_ETM_C1',
            latitude=lat,
            longitude=long,
            start_date='1995-01-01',
            end_date='2019-01-01',
            max_cloud_cover=10
        )

        entity_id = scenes[0]['entityId']
        self.ee_api.download(scene_id=entity_id, output_dir='data2')