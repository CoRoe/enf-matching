#!/usr/bin/python3

#
# Unit test cases
#
# Usage: pytest test.py
#

import pytest
import os
import datetime
import numpy as np

from griddata import GBNationalGrid, Fingrid

testdb = "/tmp/hum_test.sqlite3"


@pytest.fixture
def delete_db():
    try:
        os.remove(testdb)
    except Exception:
        pass


def progressCallback(a, b):
    pass


#
# Test set: Retrieve historical ENF values from UK National Grid and FinGrid.
#
# FIXME: FinGrid tests no longer work.
#
def test_gb_bad_date1(delete_db):
    """Verify that None is returned when asking for a year/month combination that
    is not in the database."""
    g = GBNationalGrid(testdb)
    enf = g.getEnfSeries(2000, 2, 1, progressCallback)
    assert type(enf) == tuple
    assert enf[0] is None, "Not supported year"


def not_a_test_gb_bad_date2():
    g = GBNationalGrid(testdb)
    enf = g.getEnfSeries(2000, 0, 1, progressCallback)
    assert type(enf) == tuple
    assert enf[0] is None, "Not supported year"


def xtest_gb_caching1(delete_db):
    g = GBNationalGrid(testdb)
    enf1 = g.getEnfSeries(2023, 12, 1, progressCallback)
    assert type(enf1) == tuple, "should return tuple o (data, timestamp)"
    assert enf1 is not None
    assert type(enf1[0]) == np.ndarray
    assert len(enf1[0]) == 31 * 24 * 60 * 60
    t0 = datetime.datetime.utcnow()
    enf2 = g.getEnfSeries(2023, 12, 1, progressCallback)
    assert type(enf1) == tuple, "should return tuple o (data, timestamp)"
    assert enf2[0] is not None
    assert type(enf2[0]) == np.ndarray
    assert len(enf2[0]) == len(enf1[0])
    t1 = datetime.datetime.utcnow()
    dt = t1 - t0
    assert dt.total_seconds() < 5, "Reading from DB should not take longer than 5 seconds"


def xtest_gb_caching2(delete_db):
    g = GBNationalGrid(testdb)
    enf1 = g.getEnfSeries(2015, 1, 1, progressCallback)
    assert type(enf1) == tuple, "should return tuple o (data, timestamp)"
    assert enf1[0] is not None
    assert type(enf1[0]) == np.ndarray
    assert len(enf1[0]) == 31 * 24 * 60 * 60
    t0 = datetime.datetime.utcnow()
    enf2 = g.getEnfSeries(2015, 1, 1, progressCallback)
    assert type(enf1) == tuple, "should return tuple o (data, timestamp)"
    assert enf2[0] is not None
    assert type(enf2[0]) == np.ndarray
    assert len(enf2[0]) == len(enf1[0])
    t1 = datetime.datetime.utcnow()
    dt = t1 - t0
    assert dt.total_seconds() < 5, "Reading from DB should not take longer than 5 seconds"


def xtest_fi_bad_date():
    g = Fingrid(testdb)
    enf = g.getEnfSeries(2000, 2, 1, progressCallback)
    assert type(enf) == tuple, "should return tuple o (data, timestamp)"
    assert enf[0] is None, "Not supported year"


def xtest_fingrid_caching1(delete_db):
    """Download data from the Fingrid web site; check that caching in the
    database works.

    Note: Some values in the CSV files are missing, so the overall is lower
    """
    g = Fingrid(testdb)
    enf1 = g.getEnfSeries(2017, 2, 1, progressCallback)
    assert enf1 is not None
    assert type(enf1) == tuple, "should return tuple o (data, timestamp)"
    assert type(enf1[0]) == np.ndarray
    assert len(enf1[0]) > 80000
    t0 = datetime.datetime.utcnow()
    enf2 = g.getEnfSeries(2017, 2, 1, progressCallback)
    assert enf2 is not None
    assert type(enf2) == tuple, "should return tuple o (data, timestamp)"
    assert type(enf2[0]) == np.ndarray
    assert len(enf2[0]) == len(enf1[0])
    t1 = datetime.datetime.utcnow()
    dt = t1 - t0
    assert dt.total_seconds() < 5,"Reading from DB should not take longer than 5 seconds"


def xtest_fingrid_caching2(delete_db):
    """Download data from the Fingrid web site; check that caching in the
    database works.

    Note: Some values in the CSV files are missing, so the overall is lower
    """
    g = Fingrid(testdb)
    enf1 = g.getEnfSeries(2023, 12, 1, progressCallback)
    assert enf1 is not None
    assert type(enf1) == tuple, "should return tuple o (data, timestamp)"
    assert type(enf1[0]) == np.ndarray
    assert len(enf1[0]) > 80000
    t0 = datetime.datetime.utcnow()
    enf2 = g.getEnfSeries(2023, 12, 1, progressCallback)
    assert enf2 is not None
    assert type(enf2) == tuple, "should return tuple o (data, timestamp)"
    assert type(enf2[0]) == np.ndarray
    assert len(enf2[0]) == len(enf1[0])
    t1 = datetime.datetime.utcnow()
    dt = t1 - t0
    assert dt.total_seconds() < 5,"Reading from DB should not take longer than 5 seconds"
