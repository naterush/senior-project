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
        'catalogId': 'EE'
    }
    login_url = 'https://earthexplorer.usgs.gov/inventory/json/login'
    response = requests.post(login_url, data={'jsonRequest': json.dumps(params)})
    response = response.json()
    api_key = response['data']
    print('Got API Token: ' + api_key)
    return api_key

api_token = get_auth_token(username, password)
datasetName = '201504_five_counties_pa_1ft_sp_cnir'

search_params = {
	"apiKey": api_token,
    "datasetName": datasetName
}
print("Searching datasets " + datasetName)
# response = ee_request('datasets', search_params)
response = requests.get(ee_base_url + 'search', data={'jsonRequest': json.dumps(search_params)})
print("Received datasets")
print(response)
print_json(response.json())
