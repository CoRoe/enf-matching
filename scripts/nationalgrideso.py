#!/usr/bin/python3

# Load ENF data from nationalgrideso in GB
#
# TODO: Use data type 'data' to take timezones etc. into account and to
# have a reproducible date format.


import requests
import os
import h5py
import argparse


url = "https://data.nationalgrideso.com/system/system-frequency-data/datapackage.json"
filename = "nationalgrideso.hp5"


def query_nationalgrideso(year: str, month: str):
    """ """
    timestamp = None
    ## Request execution and response reception
    print(f"Querying {url} ...")
    response = requests.get(url)
    print(f"... Status: {response.status_code}")

    ## Converting the JSON response string to a Python dictionary
    if response.ok:
        ret_data = response.json()["result"]
        try:
            csv_resource = next(
                r
                for r in ret_data["resources"]
                if r["path"].endswith(f"/f-{year}-{month}.csv")
            )
            print(f"Downloading {csv_resource['path']} ...")
            response = requests.get(csv_resource["path"])
            print(f"... Status: {response.status_code}")
            timestamp = response.text.split(os.linesep)[1].split(",")[0]
            timestamp = f"{year}-{month}-01 00:00:00"
            try:
                print("Extracting frequencies ...")
                data = [
                    float(row.split(",")[1])
                    for row in response.text.split(os.linesep)[1:-1]
                ]
            except Exception as e:
                print(e)
            return timestamp, data

        except Exception as e:
            print(e)
            return timestamp, None
    else:
        return timestamp, None

    return timestamp, None


def save_to_file(timestamp: str, datapoints: list):
    """Save a dataset from gridradar to a H5 file.

    :param timestamp: Timestamp of the first item of the dataset.
    :param datapoints: The dataset from gridradar. It is the JSON result from a HTTP query converted
    to a Python list.
    """
    # datapoints is a list of elements [frequency, timestamp], where the frequency is
    # in Hz and the timestamp has the format 'YYYY:MM:DD HH:MM:SS'. We assume that
    # the timestamps are consecutive with a spacing of 1 second.
    #
    # The dataset in the file will be named after the first timestamp in the data.

    print(f"Time stamp: {timestamp}")

    freq = [int(x * 1000) for x in datapoints]
    try:
        with h5py.File(filename, "a") as f:
            f[timestamp] = freq
            print(f.keys())
    except Exception as e:
        print(f"File {filename}:", e)


def dump_file():
    with h5py.File(filename, "a") as f:
        print(f.keys())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Query ENF data from Nation Grid in GB")
    args = parser.parse_args()

    timestamp, data = query_nationalgrideso("2014", "1")
    if data is not None:
        save_to_file(timestamp, data)
