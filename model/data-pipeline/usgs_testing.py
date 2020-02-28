import usgs
import requests
import json

username = 'ejperelmuter'
password = 'Simbacat3471'
auth_tup = (username, password)
espa_host = 'https://espa.cr.usgs.gov/api/v1/'
usgs_host = 'https://earthexplorer.usgs.gov/inventory/json/v/1.3.0/'

""" Simple way to interact with the ESPA JSON REST API """
def espa_api(endpoint, verb='get', body=None, uauth=None):
    auth_tup = (username, password)
    response = getattr(requests, verb)(espa_host + endpoint, auth=auth_tup, json=body)
    print('{} {}'.format(response.status_code, response.reason))
    data = response.json()
    if isinstance(data, dict):
        messages = data.pop("messages", None)
        if messages:
            print(json.dumps(messages, indent=4))
    try:
        response.raise_for_status()
    except Exception as e:
        print("Exception Raised: ", e)
        # print(e)
        return None
    else:
        return data

def print_json(j):
    print(json.dumps(j, indent=4))

def earthexplorer_api(endpoint='', verb='get', body=None, uauth=None):
    response = getattr(requests, verb)(usgs_host + endpoint, auth=auth_tup, json=body)
    print('{} {}'.format(response.status_code, response.reason))
    data = response.json()
    if isinstance(data, dict):
        messages = data.pop("messages", None)
        if messages:
            print(json.dumps(messages, indent=4))
    try:
        response.raise_for_status()
    except Exception as e:
        print("Exception Raised: ", e)
        # print(e)
        return None
    else:
        return data

def get_auth_token(user, pw):
    params = {
        'username': user,
        'password': pw,
        'authType': 'EROS',
    }
    login_url = 'https://earthexplorer.usgs.gov/inventory/json/login'
    response = requests.post(login_url, data={'jsonRequest': json.dumps(params)})
    response = response.json()
    api_key = response['data']
    print('Got API Token: ' + api_key)
    return api_key

api_token = get_auth_token(username, password)


params_search = {
	"datasetName": "L8",
        "spatialFilter": {
            "filterType": "mbr",
            "lowerLeft": {
                    "latitude": 44.60847,
                    "longitude": -99.69639
            },
            "upperRight": {
                    "latitude": 44.60847,
                    "longitude": -99.69639
            }
        },
        "temporalFilter": {
            "startDate": "2014-01-01",
            "endDate": "2014-12-01"
        },
	"apiKey": api_token
}
print("searching to datasets")
response = requests.post('https://earthexplorer.usgs.gov/inventory/json/v/1.4.0/datasets',
            data={'jsonRequest': json.dumps(params_search)})
            
response = response.json()

print(response)
