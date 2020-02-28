import usgs
import requests
import json

username = 'ejperelmuter'
password = 'Simbacat3471'
auth_tup = (username, password)
ee_base_url = 'https://earthexplorer.usgs.gov/inventory/json/'

def print_json(j):
    print(json.dumps(j, indent=4))

def ee_request(request_code, params):
    response = requests.post(ee_base_url + request_code, data={'jsonRequest': json.dumps(params)})
    return response

def get_auth_token(user, pw):
    params = {
        'username': user,
        'password': pw,
        'authType': 'EROS',
        "catalogId": "EE"
    }
    login_url = 'https://earthexplorer.usgs.gov/inventory/json/login'
    response = requests.post(login_url, data={'jsonRequest': json.dumps(params)})
    response = response.json()
    print_json(response)
    api_key = response['data']
    print('Got API Token: ' + api_key)
    return api_key

api_token = get_auth_token(username, password)

search_params = {
    "apiKey": api_token,
    "spatialFilter": {
        "filterType": "mbr",
        "lowerLeft": {
                "latitude": 39.93,
                "longitude": -75.20
        },
        "upperRight": {
                "latitude": 39.98,
                "longitude": -75.00
        }
    },
    "temporalFilter": {
        "startDate": "2015-01-01",
        "endDate": "2015-04-01"
    }
}
print("Searching datasets")
# response = ee_request('datasets', search_params)
response = requests.get(ee_base_url + 'search', data={'jsonRequest': json.dumps(search_params)})
print("Received datasets")
print(response)
print_json(response.json())
