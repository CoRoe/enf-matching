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
import argparse
from numpy import uint16


url = 'https://api.gridradar.net/query'
filename = 'gridradar.hp5'


def query_nationalgrideso():
    ## API query parameters
    request = {
      "metric": "frequency-ucte-median-1s",
      "format": "json",
      "ts":     "rfc3339",
      "aggr":   "1s"
    }

    ## Read token from file
    with open("gridradar-token.json") as f:
        token_data = json.__load(f)

    headers = {
        'Content-type': 'application/json',
        'Authorization': 'Bearer '+token_data['token']
    }

    ## Converting the Python dictionary to a JSON string
    json_request = json.dumps(request)

    ## Request execution and response reception
    print(f"Querying {url} ...")
    response = requests.post(url, data=json_request, headers=headers)
    print(f"... Status: {response.status_code}")

    ## Converting the JSON response string to a Python dictionary
    if response.ok:
        data = json.loads(response.content)
    else:
        data = None

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

    print(f"Time stamp: {timestamp}")

    freq = [int(x[0] * 1000) for x in datapoints]
    with h5py.File(filename, 'a') as f:
        f.create_dataset(timestamp, data=freq, dtype=uint16)
        #f[timestamp] = freq
        print(f.keys())


def dump_file():
    with h5py.File(filename, 'a') as f:
        print(f.keys())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Query ENF data from gridradar')
    parser.add_argument('command', choices=['query', 'dump'])
    args = parser.parse_args()

    if args.command == 'query':
        data = query_nationalgrideso()
        if data is not None: save_to_file(data)
    elif args.command == 'dump':
        dump_file()
