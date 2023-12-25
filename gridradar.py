#!/usr/bin/python3

# Load ENF data from gridradar.net
#
# The script needs an authorization token from gridradar.net; a 'free' account
# is sufficient to obtain this token. The free account is very limited. it is
# not possible to query frequencies from an arbitrary point in time.


import requests
import json
import pprint


url = 'https://api.gridradar.net/query'


## API query parameters
request = {
  "metric": "frequency-ucte-median-1s",
  "format": "json",
  "ts":     "rfc3339",
  "aggr":   "1s"
}


if __name__ == '__main__':

    ## Read token from file
    with open("gridradar-token.json") as f:
        token_data = json.load(f)

    headers = {
        'Content-type': 'application/json',
        'Authorization': 'Bearer '+token_data['token']
    }

    ## Converting the Python dictionary to a JSON string
    json_request = json.dumps(request)

    ## Request execution and response reception
    response = requests.post(url, data=json_request, headers=headers)

    ## Converting the JSON response string to a Python dictionary
    result_dict = json.loads(response.content)

    ## Pretty print response dictionary
    pprint.pprint(result_dict)
