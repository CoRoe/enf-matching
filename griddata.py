import datetime
import requests
from bs4 import BeautifulSoup
import io
import re
import zipfile
import py7zr
import sqlite3 as sq
import csv
import numpy as np
from scipy import signal
import struct as st
from _datetime import date


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
    """Super class"""

    def __init__(self, table_name, db_path):
        self.db_path = db_path
        self.table_name = table_name
        self.sql = sq.connect(db_path)


    def getEnfSeries(self, year, month, n_months, progressCallback):
        """Get a series of ENF values starting at a given year and month.

        :param year: The year to get the ENF series for
        :param month: The month
        :param n_months: The number of months

        :returns data: The ENF series or None if not all ENF data were available.
        :returns timestamp: UNIX timestamp of the beginning of the ENF time series.
        """

        progressCount = 0

        assert(type(year) == int and type(month) == int and month >= 1 and month <= 12 and n_months <= 12)
        # input datetime
        dt = datetime.datetime(year, month, 1, 0, 0)
        # epoch time
        epoch_time = datetime.datetime(1970, 1, 1)

        # subtract Datetime from epoch datetime
        delta = (dt - epoch_time)
        timestamp = int(delta.total_seconds())
        print("Second from epoch:", timestamp)

        # Accumulated per-month results
        total = np.empty((0,), dtype=np.uint16)
        for t in range(12*year+(month-1), 12*year+(month-1)+n_months):
            y = t // 12         # year
            m = t % 12 + 1      # month

            progressCallback(f"Querying values for {y}-{m:02} from database...", progressCount)

            # Check if ENF data are already in the database
            data = self.__query_db(y, m)
            assert data is None or type(data) == np.ndarray
            if data is not None:
                # Is in database
                total = np.append(total, data)
                progressCount += 1
            else:
                # Get the URL of the actual data file; the call is delegated to the derived,
                # grid-specific class
                progressCallback(f"ENF values for {y}-{m:02} not in database; downloading from internet...",
                                 progressCount)
                data = None
                url, daily, dec = self._getDateUrl(y, m)
                if url:
                    encodedData = self.__downloadFile(url)
                    if encodedData:
                        suffix = url.split('.')[-1]
                        if suffix == 'csv':
                            data = self.__processCsv(encodedData)
                        elif suffix == '7z':
                            data = self.__process7zData(encodedData, daily, dec)
                        elif suffix == 'zip':
                            data = self.__processZipData(encodedData, daily, dec)
                        else:
                            print(f"Unknown file type {suffix}")
                else:
                    print(f"Found no data for {y}-{m:02} at {url}")
                if data is not None:
                    assert type(data) == np.ndarray and data.dtype == np.uint16
                    total = np.append(total, data)
                    self.__save_to_db(data, y, m)
                    progressCount += 1
                else:
                    # Fail if ENF values cannot be fetched
                    return None, None

        assert type(total) == np.ndarray and total.dtype == np.uint16
        print(f"ENF series contains {total.nbytes/1000000} MB")
        return total, timestamp


    def __downloadFile(self, url):
        print(f"Querying {url}...")
        response = requests.get(url)
        print(f"... Status: {response.status_code}")
        if response.ok:
            return response.content
        else:
            return None


    def __processCsv(self, csv: bytes):
        print("Extracting frequencies from CSV file ...")
        assert type(csv) == bytes

        # Split the input data into rows and use the second item of each row.
        # The first row is supposed to contain the CSV header and is ignored.
        data = [np.uint16(float(row.split(b',')[1]) * 1000) for row in
                csv.splitlines()[1:]]
        if data is None:
            print("No data")
            return data
        else:
            print(f"{len(data)} records")
            arr = np.array(data, dtype=np.uint16)
            return arr


    def __process7zData(self, buffer, daily, decimationFactor):
        """Extract ENF values from a buffer with compressed CSV data.

        :param buffer: Buffer with compressed CSV files, in other words, a
        compressed file loaded into memory.
        :param decimationFactor: The CSV file may contain a higher temporal
        resulution that seconds.

        Assumption is that *buffer* contains one CSV file per day."""
        print("__process7zData")
        assert type(buffer) == bytes
        assert type(daily) == bool
        assert decimationFactor is None or type(decimationFactor) == int

        b = io.BytesIO(buffer)
        with py7zr.SevenZipFile(b, 'r') as archive:
            if daily:
                fn_pattern = re.compile(r"\d{4}-\d{2}-(\d{2})\.csv$")
                month_data_list = [None] * 32

                for fname, bio in archive.readall().items():
                    print(f'{fname}')
                    m = fn_pattern.search(fname)
                    if m:
                        csv = bio.read()
                        data = [np.uint16(float(row.split(b',')[1]) * 1000) for row in
                                   csv.splitlines()[1:]]
                        if decimationFactor:
                            data = signal.decimate(data, decimationFactor).astype(np.uint16)
                        month_data_list[int(m.group(1))] = data
                # month_data_list contains ENF data per day
                monthly_total = np.array([enf_value for per_day_list in
                                          month_data_list if per_day_list is not None
                                          for enf_value in per_day_list])
                print(f"Length of monthly_total is {len(monthly_total)}")
                assert type(monthly_total) == np.ndarray
                return monthly_total
            else:
                fn_pattern = re.compile(r"\d{4}-\d{2}\.csv")
                print("### not daily")


    def __processZipData(self, buffer, daily, decimationFactor):
        """Extract ENF values from a buffer with compressed CSV data.

        :param buffer: Buffer with compressed CSV files, in other words, a
        compressed file loaded into memory.
        :param decimationFactor: The CSV file may contain a higher temporal
        resulution that seconds.

        Assumption is that *buffer* contains one CSV file per day."""
        print("__process7zData")
        assert type(buffer) == bytes
        assert type(daily) == bool
        assert decimationFactor is None or type(decimationFactor) == int

        with zipfile.ZipFile(io.BytesIO(buffer)) as archive:
            if daily:
                for fname in archive.namelist():
                    print(fname)
                    month_data_list = [None] * 32
                    fn_pattern = re.compile(r"\d{4}-\d{2}-(\d{2})\.csv$")
                    m = fn_pattern.search(fname)
                    if m:
                        csv = archive.read(fname)
                        data = [np.uint16(float(row.split(b',')[1]) * 1000) for row in
                                   csv.splitlines()[1:]]
                        if decimationFactor:
                            data = signal.decimate(data, decimationFactor).astype(np.uint16)
                        month_data_list[int(m.group(1))] = data
                # month_data_list contains ENF data per day
                monthly_total = np.array([enf_value for per_day_list in
                                          month_data_list if per_day_list is not None
                                          for enf_value in per_day_list])
                print(f"Length of monthly_total is {len(monthly_total)}")
                assert type(monthly_total) == np.ndarray
                return monthly_total
            else:
                print("Not supported")


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
            l = np.array(st.unpack(fmt, blob), dtype=np.uint16)
            return l
        else:
            print("... not found")
            return None


    def __save_to_db(self, dataset, year, month):
        assert type(dataset) == np.ndarray and dataset.dtype == np.uint16
        try:
            db_key = str(f"{year:04}-{month:02}")
            print(f"Inserting array of len {len(dataset)} at {db_key}...")
            rc = self.cursor.execute(f"INSERT INTO {self.table_name} (date, frequencies) VALUES (?, ?)",
                                     (db_key, dataset))
            rc = self.sql.commit()
        except sq.Error as e:
            print(e)


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


    def getDateRange(self):
        url ='https://data.fingrid.fi/en/dataset/frequency-historical-data'
        pattern = re.compile(r"(\d+)-(\d+)\.(zip|7z)")
        fromDate = (9999, 9999)
        toDate = (0, 0)

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
                m = pattern.search(r['href'])
                if m:
                    #print(m.group(1), m.group(2))
                    d = (int(m.group(1)), m.group(2))
                    if d < fromDate: fromDate = d
                    if d > toDate: toDate = d
        return f"{fromDate[0]}-{fromDate[1]}", f"{toDate[0]}-{toDate[1]}"


    def _getDateUrl(self, year: int, month: int):
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
                if url.endswith(f"{year:4}-{month:02}.zip") or url.endswith(f"{year:4}-{month:02}.7z"):
                    return url, True, 10
        return None, None, None


    def __processCSV_unused(self, fp):
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


    def getDateRange(self):
        url = 'https://data.nationalgrideso.com/system/system-frequency-data/datapackage.json'
        pattern = re.compile(r"(\d+)-(\d+)\.csv$")
        fromDate = (9999, 9999)
        toDate = (0, 0)

        ## Request execution and response reception
        print(f"Querying {url} ...")
        response = requests.get(url)
        print(f"... Status: {response.status_code}")

        ## Converting the JSON response string to a Python dictionary
        if response.ok:
            ret_data = response.json()['result']['resources']
            for d in ret_data:
                m = pattern.search(d['path'])
                if m:
                    #print(m.group(1), m.group(2))
                    date = (int(m.group(1)), int(m.group(2)))
                    if date < fromDate: fromDate = date
                    if date > toDate: toDate = date

        return f"{fromDate[0]}-{fromDate[1]:02}", f"{toDate[0]}-{toDate[1]:02}"


    def _getDateUrl(self, year, month):
        """
        Download ENF historical data from the GB National Grid database.

        :param location: ignored
        :param year: year
        :param month: month
        :returns np.array with the ENF values or None if not found. ENF values
        are the frequency in mHz.
        """
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
                                    if r['path'].endswith(f"-{year}-{month}.csv"))
                return csv_resource['path'], False, None
            except Exception as e:
                print(e)
                return None, False, None
        print("End of loadGridEnf")
        return None, False, None


if __name__ == '__main__':
    db_path = "/tmp/hum.sqlite"

    g = GridDataAccessFactory.getInstance('GB', db_path)
    g.getEnfSeries(1900, 11)
    g.getEnfSeries(2023, 11)

    g = GridDataAccessFactory.getInstance('FI', db_path)
    g.getEnfSeries(2015, 1)     # ZIP compressed
    g.getEnfSeries(2023, 11)    # 7z compressed
