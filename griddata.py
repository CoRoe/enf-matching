#!/usr/bin/python3

import datetime
import requests
from bs4 import BeautifulSoup
import io
import os
import re
import zipfile
import sqlite3 as sq
import csv
import numpy as np
from scipy import signal
import struct as st


#
# Database is hum.sqlite
#
# Has a table for each ENF data source; e.g. 'fingrid'
#
# Each table has one row per month. Key has the format yyyy-mm, the data is an
# array of uint16 with one entry per second.
#


class GridDataAccessFactory():
    
    locations = ['GB', 'FI']

    @classmethod
    def getInstance(cls, location, database_path):
        if location == 'GB':
            return GBNationalGrid(database_path)
        elif location == 'FI':
            return Fingrid(database_path)
        else:
            return None
    
    @classmethod
    def enumLocations(cls):
        for l in GridDataAccessFactory.locations:
            yield l


class GridDataAccess():

    def __init__(self, table_name, db_path):
        self.db_path = db_path
        self.table_name = table_name
        self.sql = sq.connect(db_path)
        pass


    def getEnfSeries(self, year, month):
        """Get a series of ENF values for a given year and month.
        """
        assert(type(year) == int and type(month) == int and month >= 1 and month <= 12)

        # Check if ENF data are already in the database
        data = self.__query_db(year, month)
        if data is not None:
            # Is in database
            return data
        else:
            # Download from internet; the call is delegated to the derived,
            # grid-specific class
            data = self._downloadFromInternet(year, month)
            if data is not None:
                self.__save_to_db(data, year, month)
            return data


    def __query_db(self, year, month):
        """Query the database for blob list of ENF at blob given date.

        :param year: Year
        :param month:
        :returns: List of ENF values, one per second, or None.
        """
        db_key = f"{year:04}-{month:02}"
        print(f"Querying {db_key} from {self.table_name}")
        res = self.cursor.execute(f"SELECT date, frequencies from {self.table_name} WHERE date IS \"{db_key}\"")
        res_list = list(res)
        if len(res_list) == 1:
            print("...OK")
            blob = res_list[0][1]
            fmt = f"{int(len(blob)/2)}H"
            l = np.array(st.unpack(fmt, blob))
            return l
        else:
            print("... not found")
            return None


    def __save_to_db(self, dataset, year, month):
        try:
            db_key = str(f"{year:04}-{month:02}")
            print(f"Inserting array of len {len(dataset)} at {db_key}...")
            rc = self.cursor.execute(f"INSERT INTO {self.table_name} (date, frequencies) VALUES (?, ?)",
                                     (db_key, dataset))
            rc = self.sql.commit()
        except sq.Error as e:
            print(e)


    def getFromDate(self):
        """Return source, country, time resolution, time zone, from, to."""
        # TODO: Implement
        return "2015-01"

    def getToDate(self):
        """Return source, country, time resolution, time zone, from, to."""
        # TODO: Implement
        return "2020-01"


# Zeilen der Form
# <a class="heading" href="/en/dataset/frequency-historical-data/resource/65885c01-8199-4c8a-9679-c0002b894b9a" title="Taajuusmittausdata 2015-01">
# extrahieren
#
# The ZIP file contains 1 CSV file per day; filenames are like 2015-10-01.csv, 2015-10-02.csv, and so on.
#
# Each ZIP file has the format
#
# Time,Value
# 2015-01-01 00:00:00.000,50.103
# 2015-01-01 00:00:00.100,50.103
# 2015-01-01 00:00:00.200,50.104
#

class Fingrid(GridDataAccess):
    table_name = 'Fingrid'

    def __init__(self, db_path):
        super(Fingrid, self).__init__(Fingrid.table_name, db_path)

        self.cursor = self.sql.cursor()
        rc = self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {Fingrid.table_name} (date TEXT PRIMARY KEY, frequencies BLOB)")
        rc = self.sql.commit()
        pass


    def _downloadFromInternet(self, year, month):
        """Download ENF data from the internet.
        :param year: year
        :param month: Month
        :returns: Sequence of ENF values
        """
        zipurl = self.__getURL(year, month)
        print("Downloading zip file from", zipurl)
        if zipurl:
            response = requests.get(zipurl)
            print(f"... Status: {response.status_code}")
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                data = self.__data_from_open_zip_file(zf, year, month)
                return data
        else:
            return None


    def getEnfSeriesFromZipFile(self, zip_name, year: int, month: int):
        """Instead of dowloading a ZIP file open one already stored in file
        system and process it.

        :param zip_name: Path of the ZIP file.
        :param year:
        :param month: 1..12
        """
        assert(month >= 1 and month <= 12)

        zf = zipfile.ZipFile(zip_name)
        return self.__data_from_open_zip_file(zf, year, month)


    def __data_from_open_zip_file(self, zf, year, month):
        """Extract data from an open ZIP file.

        :param zf: The open ZIP file; it refers to either a buffer with
        downloaded data or a disk file.
        :param year:
        :param month:

        The ZIP file contains a number of CSV encoded files, one for each day.
        Extract the ENF data from each file and combine them into one large
        array.
        """
        fn_pattern = re.compile(r"\d{4}-\d+-\d+$")
        month_data_list = [None] * 31

        print ("Uncompressing and reading data... ")

        for csv_file_info in zf.infolist():
            print(csv_file_info.filename)
            key = csv_file_info.filename[:-4]            # Strip the suffix '.csv'
            with zf.open(csv_file_info.filename, 'r') as csv_file:
                # Check if the filename is like yyyy-mm-dd
                if fn_pattern.match(key):
                    dataset = self.__processCSV(csv_file)
                    day = int(key[8:10])
                    d_dataset = signal.decimate(dataset, 10).astype(np.uint16)
                    if len(d_dataset) != 86400:
                        print(f"Warning: {csv_file_info.filename} has {len(d_dataset)} elements")
                    # assert len(d_dataset) == 86400
                    #print(len(dataset), len(d_dataset))
                    month_data_list[day - 1] = d_dataset

        # month_data_list contains ENF data per day
        monthly_total = np.array([enf_value for per_day_list in
                                  month_data_list if per_day_list is not None
                                  for enf_value in per_day_list])
        print(f"Length of monthly_total is {len(monthly_total)}")
        assert type(monthly_total) == np.ndarray
        return monthly_total


    def __getURL(self, year: int, month: int):
        assert type(year) == int and type(month) == int, "should be integers"
        url ='https://data.fingrid.fi/en/dataset/frequency-historical-data'
        print(f"Querying {url}...")
        response = requests.get(url)
        print(f"... Status: {response.status_code}")
        if response.ok:
            ret_data = response.text
            #print(ret_data)
            soup = BeautifulSoup(ret_data, "html.parser")
            res = soup.find_all('a', class_='resource-url-analytics')
            for r in res:
                #print(r['href'])
                url = r['href']
                if url.endswith(f"{year:4}-{month:02}.zip"):
                    return url
        return None


    def __processCSV(self, fp):
        """Process a CSV file.

        :param fp: The open CSV file.
        :returns: An array of uint16 wth the grid frequencies.
        """
        csvReader = csv.reader(io.TextIOWrapper(fp, 'utf-8'))
        a = [np.uint16(float(row[1])*1000) for row in csvReader
             if row[0] != 'Time']
        return a


class GBNationalGrid(GridDataAccess):
    table_name = 'GBNationalGrid'

    def __init__(self, db_path):
        super(GBNationalGrid, self).__init__(GBNationalGrid.table_name, db_path)

        self.cursor = self.sql.cursor()
        rc = self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {GBNationalGrid.table_name} (date TEXT PRIMARY KEY, frequencies BLOB)")
        rc = self.sql.commit()
        pass


    def _downloadFromInternet(self, year, month):
        """
        Download ENF historical data from the GB National Grid database.

        :param location: ignored
        :param year: year
        :param month: month
        :returns np.array with the ENF values or None if not found. ENF values
        are the frequency in mHz.
        """
        arr = None
        url = 'https://data.nationalgrideso.com/system/system-frequency-data/datapackage.json'

        ## Request execution and response reception
        print(f"Querying {url} ...")
        response = requests.get(url)
        print(f"... Status: {response.status_code}")

        ## Converting the JSON response string to a Python dictionary
        if response.ok:
            ret_data = response.json()['result']
            try:
                csv_resource = next(r for r in ret_data['resources']
                                    if r['path'].endswith(f"/f-{year}-{month}.csv"))
                print(f"Downloading {csv_resource['path']} ...")
                response = requests.get(csv_resource['path'])
                print(f"... Status: {response.status_code}")
                try:
                    print("Extracting frequencies ...")
                    data = [np.uint16(float(row.split(',')[1]) * 1000) for row in
                            response.text.split(os.linesep)[1:-1]]
                    if data is None:
                        print("No data")
                    else:
                        print(f"{len(data)} records")
                        arr = np.array(data)
                except Exception as e:
                    print(e)
            except Exception as e:
                print(e)
        print("End of loadGridEnf")
        return arr


if __name__ == '__main__':
    db_path = "/tmp/hum.sqlite"

    g = GridDataAccessFactory.getInstance('FI', db_path)
    g.getEnfSeries(2017, 2)
