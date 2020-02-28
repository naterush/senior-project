# This file demonstrates the capabilities of the landsat API
# URL: https://pypi.org/project/landsatxplore/

import landsatxplore.api
from landsatxplore.earthexplorer import EarthExplorer
import json
# import urllib.request

username = 'ejperelmuter'
password = 'Simbacat3471'

# Initialize a new API instance and get an access key
# landsat_api = landsatxplore.api.API(username, password)

ee_api = EarthExplorer(username, password)
ee_api.download(scene_id='LE71960461999181EDC00', output_dir='data/')
ee_api.logout()

# Request
scenes = landsat_api.search(
    dataset='LANDSAT_ETM_C1',
    latitude=19.53,
    longitude=-1.53,
    start_date='1995-01-01',
    end_date='2019-01-01',
    max_cloud_cover=10)

num = str(len(scenes))
print(num + ' scenes found.')
print(json.dumps(scenes[0], indent=4))



for scene in scenes:
    print(json.dumps(scene, indent=4))
    print(scene['acquisitionDate'])

landsat_api.logout()
