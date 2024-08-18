# a script to collect ENF data from the European live ENF measurement
# https://www.mainsfrequency.com/
# the code they have is JS, check mains.js

import requests
import random
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import time
import tempfile
import argparse


hostname = "netzfrequenzmessung.de"

description = f"""
Query net frequency data from {hostname} and saves them to a file.

Basic usage is\n

    'python3 get_live_data'

Press the interrupt key CTRL-C to terminate the program.
"""

# the numbers 310_000 and -31 always work, but if you send too many requests,
# they will tell you so...  so this function gives a random one of the two
def get_c():
    opts = [-31, 310_000]
    return str(random.choice(opts))

def get_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def get_enf_data():
    # Placeholder data to test if rate limited
    # return {
    #     "frequency": 50.0,
    #     "time": "2024-08-12 18:05:20",
    #     "phase": 276.4,
    #     "d": 7.0,
    #     "id": get_id()
    # }f'Query net frequency data from {hostname} and saves them to a file')
    url = "https://netzfrequenzmessung.de:9081/frequenz02c.xml?c=" + get_c()
    ip = get_ip()
    headers = {
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Host": "www.mainsfrequency.com",
        "Referer": "https://www.mainsfrequency.com/", # trust me bro
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": 'Firefox',
        "Forwarded": "for=" + ip,
        "X-Forwarded-For": ip,
    }
    response = requests.get(url, headers=headers)
    data = response.text

    try:
        xml = ET.fromstring(data)
        freq = xml.find("f2").text
        time = xml.find("z").text
        phase = xml.find("p").text
        d = xml.find("d").text # I don't know what d is, but it's there so it's (probably) important

        data = {
            "time": datetime.strptime(time.strip(), '%d.%m.%Y %H:%M:%S'),
            "frequency": float(freq),
            #"phase": float(phase),
            #"d": float(d),
        }

        return data

    except Exception as e:
        # Typically happens because we've been rate limited... I don't know how to handle this
        print(e)
        print(data)
        return None


def data_loop(filename, limit):
    """Get the network frequency from a host and save them as CSV file.

    @param filenbame: Name ot the CSV file.
    @param Number of entries to query.
    """
    print(f"Querying ENF data from {hostname} ...")
    data = []
    try:
        for _ in range(limit):
            record = get_enf_data()
            if record:
                data.append(record)
                time.sleep(1 - datetime.now().microsecond / 1_000_000)
            else:
                # If we get "too many requests", we can just wait a bit and it tends to start working.
                time.sleep(1)
    except KeyboardInterrupt:
        print('Interrupted')

    if len(data) > 0:
        df = pd.DataFrame(data)
        # print(df)

        # FIXME: Handle the case that no data were collected
        df.index = df['time']
        del df['time']

        # Resample at 1 second interval; this will insert NaNs for missing
        # timestamps
        resampled = df.resample('s').mean()

        # Interpolate missing data
        interpolated = resampled.interpolate()

        # Save to CSV file
        interpolated.to_csv(filename)
        print(f"... saved to {filename}")
    else:
        print("... no data collected")


def get_default_filename():
    now = datetime.utcnow()
    fn = now.strftime("%Y-%m-%dT%H:%M:%S.csv")
    tmpdir = tempfile.gettempdir()
    return tmpdir + '/' + fn


def main():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-o', default=get_default_filename(), help='Output filename')
    parser.add_argument('-c', '--count', type=int, default=100000)
    args = parser.parse_args()

    data_loop(args.o, args.count)


if __name__ == "__main__":
    main()
