#!/usr/bin/python3

# Load ENF data from gridradar.net
#
# The script needs an authorization token from gridradar.net; a 'free' account
# is sufficient to obtain this token. The free account is very limited. it is
# not possible to query frequencies from an arbitrary point in time.
#
# The authentication token is expected in a file 'gridradar-token.json'; the file
# format is
#
#    {
#       "token": token-obtained-from-gridradar
#    }
#
# https://docs.h5py.org/en/stable/quick.html


import requests
import json
import h5py


url = 'https://api.gridradar.net/query'
filename = 'gridradar.hp5'


def query_gridradar():
    ## API query parameters
    request = {
      "metric": "frequency-ucte-median-1s",
      "format": "json",
      "ts":     "rfc3339",
      "aggr":   "1s"
    }

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
    data = json.loads(response.content)

    return data


def save_to_file(d: list):
    """ Save a dataset from gridradar to a H5 file.

    :param: d: The dataset from gridradar. It is the JSON result from a HTTP query converted
    to a Python list.
    """
    # datapoints is a list of elements [frequency, timestamp], where the frequency is
    # in Hz and the timestamp has the format 'YYYY:MM:DD HH:MM:SS'. We assume that
    # the timestamps are consecutive with a spacing of 1 second.
    #
    # The dataset in the file will be named after the first timestamp in the data.
    datapoints = d[0]['datapoints']
    timestamp = datapoints[0][1]

    freq = [int(x[0] * 1000) for x in datapoints]
    f = h5py.File(filename, 'w')
    # Write dataset
    f.create_dataset(timestamp, data=freq)
    # Close file and write data to disk. Important!
    f.close()


if __name__ == '__main__':

    data = query_gridradar()
    save_to_file(data)
