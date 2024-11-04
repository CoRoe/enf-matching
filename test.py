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
from enf import AudioClipEnf, GridEnf, VideoClipEnf

testdb = "/tmp/hum_test.sqlite3"
wavef = 'samplemedia/001.wav'
gridwavef = 'samplemedia/71000_ref.wav'
videoclip_rs = 'fluorescent.mp4'


@pytest.fixture
def delete_db():
    try:
        os.remove(testdb)
    except Exception:
        pass


def progressCallback(a, b):
    pass


def plotCallback(x, y):
    pass


#
# Test set video files
#
def test_video_clip_rs():
    clip = VideoClipEnf()
    clip.getVideoProperties(videoclip_rs)
    clip.loadVideoFileRollingShutter(videoclip_rs, 25)


#
# Test set audio files; tets uses a specific file with known properties
#
def test_audio_file_loading():
    """Verify that a WAV file can be loaded."""
    clip = AudioClipEnf()
    clip.loadAudioFile(wavef)
    assert clip.getDuration() == 482
    assert clip.sampleRate() == 400


def test_audio_clip_analysis():
    """Analyse a clip."""
    clip = AudioClipEnf()
    clip.loadAudioFile(wavef, fs=8000)

    # Extract ENF values from the clip without removing outliers
    clip.makeEnf(50, 0.2, 2)
    t, enf = clip.getEnf()
    assert enf.shape == (484,)
    assert t.shape == (484,)
    assert np.max(enf) <= 50186
    assert np.min(enf) >= 49824
    t, enfs = clip.getEnfs()
    assert t is None and enfs is None

    clip.outlierSmoother(3.0, 5)
    t, enf = clip.getEnf()
    assert enf.shape == (484,)
    t, enfs = clip.getEnfs()
    assert enfs.shape == (484,)

    # Compute spectrogram
    f, t, Sxx = clip.makeSpectrogram()
    assert f.shape == (510,)
    assert t.shape == (551,)
    assert Sxx.shape == (510, 551)

    # The region must be defined
    rgn = clip.getENFRegion()
    assert rgn == (0, 482)


def test_grid_analysis_wavef():
    grid = GridEnf(testdb)
    grid.loadAudioFile(gridwavef)
    grid.makeEnf(50, 0.2, 2)
    t, enf = grid.getEnf()
    assert t.shape == (15366,)
    assert enf.shape == (15366,)
    assert np.max(enf) <= 50186
    assert np.min(enf) >= 49810
    assert grid.ENFavailable()


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


def test_gb_caching1(delete_db):
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


def test_gb_caching2(delete_db):
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
