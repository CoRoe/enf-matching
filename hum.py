#!/usr/bin/python3

# https://www.pythonguis.com/tutorials/plotting-pyqtgraph/'


import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
                             QVBoxLayout, QLineEdit, QFileDialog, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QCheckBox,
                             QComboBox, QSpinBox, QTabWidget, QDoubleSpinBox,
                             QMenuBar, QAction, QDialog, QMessageBox,
                             QDialogButtonBox, QProgressDialog)
from PyQt5.Qt import Qt

from scipy import signal, fft
import wave
import numpy as np
import datetime
import os
import subprocess
import json
from griddata import GridDataAccessFactory
import pandas as pd


def butter_bandpass_filter(data, locut, hicut, fs, order):
    """Passes input data through a Butterworth bandpass filter. Code borrowed from
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

    :param data: list of signal sample amplitudes
    :param locut: frequency (in Hz) to start the band at
    :param hicut: frequency (in Hz) to end the band at
    :param fs: the sample rate
    :param order: the filter order
    :returns: list of signal sample amplitudes after filtering
    """
    nyq = 0.5 * fs
    low = locut / nyq
    high = hicut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')

    return signal.sosfilt(sos, data)


def stft(data, fs):
    """Perform a Short-time Fourier Transform (STFT) on input data.

    :param data: list of signal sample amplitudes
    :param fs: the sample rate
    :returns: tuple of (array of sample frequencies, array of segment times, STFT of input).
    This is the same return format as scipy's stft function.
    """
    window_size_seconds = 16
    nperseg = fs * window_size_seconds
    noverlap = fs * (window_size_seconds - 1)
    f, t, Zxx = signal.stft(data, fs, nperseg=nperseg, noverlap=noverlap)
    return f, t, Zxx


def enf_series(data, fs, nominal_freq, freq_band_size, harmonic_n=1):
    """Extracts a series of ENF values from `data`, one per second.

    :param data: list of signal sample amplitudes
    :param fs: the sample rate
    :param nominal_freq: the nominal ENF (in Hz) to look near
    :param freq_band_size: the size of the band around the nominal value in which to look for the ENF
    :param harmonic_n: the harmonic number to look for
    :returns: a list of ENF values, one per second
    """
    # downsampled_data, downsampled_fs = downsample(data, fs, 300)
    downsampled_data, downsampled_fs = (data, fs)

    locut = harmonic_n * (nominal_freq - freq_band_size)
    hicut = harmonic_n * (nominal_freq + freq_band_size)

    filtered_data = butter_bandpass_filter(downsampled_data, locut, hicut,
                                           downsampled_fs, order=10)

    f, t, Zxx = stft(filtered_data, downsampled_fs)

    def quadratic_interpolation(data, max_idx, bin_size):
        """
        https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        """
        left = data[max_idx-1]
        center = data[max_idx]
        right = data[max_idx+1]

        p = 0.5 * (left - right) / (left - 2*center + right)
        interpolated = (max_idx + p) * bin_size

        return interpolated

    bin_size = f[1] - f[0]

    max_freqs = []
    for spectrum in np.abs(np.transpose(Zxx)):
        max_amp = np.amax(spectrum)
        max_freq_idx = np.where(spectrum == max_amp)[0][0]

        max_freq = quadratic_interpolation(spectrum, max_freq_idx, bin_size)
        max_freqs.append(max_freq)

    return {
        'downsample': {
            'new_fs': downsampled_fs,
        },
        'filter': {
            'locut': locut,
            'hicut': hicut,
        },
        'stft': {
            'f': f,
            'tablew': t,
            'Zxx': Zxx,
        },
        'enf': [f/float(harmonic_n) for f in max_freqs],
    }


class EnfModel():
    """Model an ENF series.

    The class has two slightly different purposes: Model an ENF series as read
    from grid databases on the internet, and model an ENF series derived from
    an audio clip. For the latter purpose, it contains methods to analyse the
    audio: Fourier transforms, filter, outlier removal, etc.

    When used for grid Data, the ENF series is cached in an sqlite database;
    the path of this database must be passed a parameter to the constructor,
    It may also be changed later.

    All time series are internally handled as Numpy arrays with the shape (N,
    0) where N is the number of items of the array. The timestamp of the first
    element of a time series is kept in an instance variable; the following
    elements are evenly spaced by 1 second.

    """

    def __init__(self, databasePath):
        """Create an empty ENF series.

        :param dataBasePath: File system path where the database will be
        cached.
        """
        self.data = None
        self.enf = None
        self.enfs = None
        self.databasePath = databasePath
        self.fft_freq = None
        self.fft_ampl = None
        self.timestamp = None
        self.aborted = False


    def setDatabasePath(self, path):
        self.databasePath = path


    def fromWaveFile(self, fpath):
        """Loads wave_buf .wav file and computes ENF and SFT.

        :param fpath: the path to __load the file from rate)

        On exit, self.data is an Numpy array with the samples of the loaded
        audio recording ('clip').  If the WAV file has a sampling rate above
        8,000 Hz it is decimated down to 8,000 Hz. The function throws an
        exception if the original sampling rate is not a multiple of 8,000.

        """
        # TODO: Check big and low endianness
        with wave.open(fpath) as wav_f:
            self.fs = wav_f.getframerate()
            self.n_frames = wav_f.getnframes()
            if self.fs > 8000:
                ds_factor = int(self.fs / 8000)
                assert(ds_factor * 8000 == self.fs)
                self.data = None

                # Read chunks, downsample them
                wav_buf = wav_f.readframes(1000000)
                while len(wav_buf) > 0:
                    #print(len(wav_buf))
                    nw = signal.decimate(np.frombuffer(wav_buf, dtype=np.int16), ds_factor)
                    #print("After decimation:", len(nw))
                    if self.data is not None:
                        self.data = np.append(self.data, nw)
                    else:
                        self.data = nw
                    wav_buf = wav_f.readframes(1000000)
                self.fs = int(self.fs / ds_factor)
            else:
                # Read entire file into buffer
                wav_buf = wav_f.readframes(wav_f.getnframes())
                self.data = np.frombuffer(wav_buf, dtype=np.int16)

            assert(type(self.data) == np.ndarray)
            self.clip_len_s = int(self.n_frames / self.fs)
            print(f"File {fpath}: Sample frequency {self.fs} Hz, duration {self.clip_len_s} seconds")

            # Use current time as timestamp
            self.timestamp = int(datetime.datetime.now().timestamp())


    def makeEnf(self, nominal_freq, freq_band_size, harmonic):
        """Creates an ENF series from the sample data.

        :param: nominal_freq: Nominal grid frequency in Hz; usually 50 or 60 Hz
        :param: freq_band_size: Size of the frequency band around *nominal_freq* in Hz
        :param: harmonic:

        The method takes self.data (the samples of the audio recording) and
        computes self.enf (the series of frequencies of the 50 or 60 Hz hum of
        the recording.)

        """
        assert(self.data is not None)

        self.nominal_freq = nominal_freq
        self.freq_band_size = freq_band_size
        self.harmonic = harmonic
        enf_output = enf_series(self.data, self.fs, nominal_freq,
                                freq_band_size, harmonic_n=harmonic)

        # stft is the Short-Term Fourier Transfrom of the audio file, computed
        # per second.
        # self.stft = enf_output['stft']

        # ENF are the ENF values
        enf = [int(e * 1000) for e in enf_output['enf']]
        self.enf = np.array(enf)
        assert type(self.enf) == np.ndarray


    def loadGridEnf(self, location, year: int, month: int):
        """ Load the grid ENF values from a database.

        :param location: The name/location of the grid
        :param year: The year
        :param month: The number of the month (1 = Jan, 2 = Feb, ...)
        """
        assert(self.databasePath)
        assert(type(year) == int and year > 1970)
        assert(type(month) == int and month >= 1 and month <= 12)
        assert location != 'Test', "Handled elsewhere"
        data_source = GridDataAccessFactory.getInstance(location, self.databasePath)
        self.enf, self.timestamp = data_source.getEnfSeries(year, month)
        assert self.enf is None or type(self.enf) == np.ndarray
        assert type(self.timestamp == int)


    def makeFFT(self):
        """ Compute the spectrum of the original audio recording.

        :param: self.data: sample data of the audio file
        :param: self.fs: sample frequency
        :returns: Tuple (frequencies, amplitudes)
        """
        # https://docs.scipy.org/doc/scipy/tutorial/fft.html#d-discrete-fourier-transforms
        # Result is complex.
        assert(self.data is not None)

        spectrum = fft.fft(self.data)
        self.fft_freq = fft.fftfreq(len(spectrum), 1.0 / self.fs)
        self.fft_ampl = np.abs(spectrum)

        return self.fft_freq, self.fft_ampl


    def getMatchingSteps(self, gridModel):
        """Return the number of number of iterations. Usefull for a progress
        indicator."""
        n_steps = len(gridModel.getENF()) - len(self.enf) + 1
        return n_steps


    def matchPearson(self, ref, progressCallback):
        """Given a reference clip with ENF values find the best fit with the
        own ENF values.

        :param ref: The ENF series of the grid
        :param progressCallback: Function to be called once in a while
        to signal the progress of the processing.
        :returns: The index into the reference series of thebets match; the
        correlation at that index, an array of correlations for all possible
        indices.

        No -- the timestamp of the best match.

        The method computes the Pearson correlation between the ENF values in
        the clip and the grid.

        See: https://realpython.com/numpy-scipy-pandas-correlation-python/
        https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        """
        assert(type(ref) == EnfModel)

        self.aborted = False

        def step_enum(steps, progressCallback):
            for n in range(steps):
                if n % 1000 == 0:
                    progressCallback(n)
                yield n

        def canceled():
            if self.aborted:
                raise StopIteration

        print(f"Start Pearson correlation computation: {datetime.datetime.now()} ...")
        ref_enf = ref.getENF()
        timestamp = ref.getTimestamp()
        print(f"Len ref_enf: {len(ref_enf)}, len(enf): {len(self.enf)}")

        # If a smoothed version is available then use it
        if self.enfs is not None:
            enf = self.enfs
        else:
            enf = self.enf

        n_steps = len(ref_enf) - len(enf) + 1
        try:
            corr = [np.corrcoef(ref_enf[step:step+len(enf)], enf)[0][1]
                    for step in step_enum(n_steps, progressCallback)
                    if not canceled()]
        except StopIteration:
            print("Cancelled...")
        if self.aborted:
            return None, None, None
        else:
            max_index = np.argmax(corr)
            print(f"End Pearson correlation computation {datetime.datetime.now()} ...")
            return timestamp + max_index, corr[max_index], corr


    def matchEuclidianDist(self, ref, progressCallback):
        """Given a reference clip with ENF values find the best fit with the
        own ENF values.

        :param ref: The ENF series of the grid
        :param progressCallback: Function to be called once in a while
        to signal the progress of the processing.
        :returns: The index into the reference series of thebets match; the
        correlation at that index, an array of correlations for all possible
        indices.

        The method computes the Euclidian distance between the ENF values in
        the clip and the grid.

        See: https://www.geeksforgeeks.org/python-distance-between-collections-of-inputs/

        The methond constantly monitors self.aborted and stops if its value is True.
        """

        # Apparently the StopIteration exception can only be raised from within
        # the condition function. Unfortunate, because two functions are needed
        # in the list comprehension.
        self.aborted = False

        def step_enum(steps, progressCallback):
            for n in range(steps):
                if n % 1000 == 0:
                    progressCallback(n)
                yield n

        def canceled():
            if self.aborted:
                raise StopIteration

        assert(type(ref) == EnfModel)

        print(f"Start Euclidian correlation computation: {datetime.datetime.now()} ...")
        ref_enf = ref.getENF()
        timestamp = ref.getTimestamp()

        # If a smoothed version is available then use it
        if self.enfs is not None:
            enf = self.enfs
        else:
            enf = self.enf

        n_steps = len(ref_enf) - len(enf) + 1
        progressCallback(0)
        try:
            mse = [((ref_enf[step:step+len(enf)] - enf) ** 2).mean()
                    for step in step_enum(n_steps, progressCallback)
                    if not canceled()]
            #corr = [spatial.distance.cdist([ref_enf[step:step+len(enf)], enf],
            #                               [ref_enf[step:step+len(enf)], enf],
            #                               'sqeuclidean')[0][1] for step in step_enum(n_steps, progressCallback)
            #                                if not canceled()]
        except StopIteration:
            print("...canceled")
        if self.aborted:
            return None, None, None
        else:
            # Normalise
            corr = mse / np.sqrt(len(mse))
            min_index = np.argmin(corr)
            print(f"End Euclidian correlation computation {datetime.datetime.now()} ...")
            progressCallback(n_steps)
            return timestamp + min_index, corr[min_index], corr


    def matchConv(self, ref, progressCallback):
        """Compute correlation between clip ENF and grid ENF."""
        print("matchConv")
        grid_freqs = ref.getENF()
        if self.enfs is not None:
            enf = self.enfs
        else:
            enf = self.enf
        n_steps = len(grid_freqs) - len(enf) + 1
        timestamp = ref.getTimestamp()

        progressCallback(0)
        xcorr = signal.correlate(
            grid_freqs-np.mean(grid_freqs),
            enf-np.mean(enf),
            mode='same')
        max_index = np.argmax(xcorr)
        ref_normalization = pd.Series(grid_freqs).rolling(self.clip_len_s, center=True).std()
        signal_normalization = np.std(enf)
        xcorr_norm = xcorr/ref_normalization/signal_normalization/self.clip_len_s
        progressCallback(n_steps)
        return timestamp + max_index, xcorr_norm[max_index], xcorr_norm


    def outlierSmoother(self, threshold, win):
        """Find outliers in the ENF values replace them with the median of
        neighbouring values.

        :param threshold: Values with threshold times the median deviation are
        smoothed

        :param win:
        :param self.enf: ENF data generated previous step
        :param self.enfs: Smoothed ENF data
        """
        x_corr = np.copy(self.enf)
        d = np.abs(self.enf - np.median(self.enf))
        mdev = np.median(d)
        print(f"Deviation median: {mdev}")
        idxs_outliers = np.nonzero(d > threshold*mdev)[0]
        for i in idxs_outliers:
            if i-win < 0:
                x_corr[i] = np.median(np.append(self.enf[0:i], self.enf[i+1:i+win+1]))
            elif i+win+1 > len(self.enf):
                x_corr[i] = np.median(np.append(self.enf[i-win:i], self.enf[i+1:len(self.enf)]))
            else:
                x_corr[i] = np.median(np.append(self.enf[i-win:i], self.enf[i+1:i+win+1]))
        self.enfs = x_corr

    def getENF(self):
        return self.enf


    def getENFs(self):
        return self.enfs


    def getData(self):
        return self.data


    def getFFT(self):
        return self.fft_freq, self.fft_ampl


    def getDuration(self):
        """ Length of the clip in seconds."""
        return self.clip_len_s


    def sampleRate(self):
        """ Return the sample rate in samples / second. """
        return self.fs


    def getTimestamp(self):
        return self.timestamp


    def onCanceled(self):
        """Handles the 'cancel' signal from a QProgressDialog.

        Sets the instance variable aborted. Lengthy operations check this
        flag and stop when it is set.
        """
        self.aborted = True


class HumView(QMainWindow):
    """ Display ENF analysis and result."""

    def __init__(self):
        """Initialize variables and create widgets and menu."""
        super().__init__()
        self.clip = None
        self.grid = None
        self.settings = Settings()
        self.databasePath = self.settings.databasePath()

        self.enfAudioCurve = None     # ENF series of loaded audio file
        self.enfAudioCurveSmothed = None
        self.clipSpectrumCurve = None # Fourier transform of loaded audio file
        self.enfGridCurve = None      # ENF series of grid
        self.correlationCurve = None  # Correlation of ENF series of audio
                                      # clip and grid

        self.__createWidgets()
        self.__createMenu()


    def __createWidgets(self):
        """Create widgets including curves and legends for the plot widgets."""
        widget = QWidget()
        self.setWindowTitle("Hum")

        # Define layouts
        main_layout = QVBoxLayout()
        audio_area = QGridLayout()
        audio_group = QGroupBox("Audio")
        audio_group.setLayout(audio_area)
        analyse_group = QGroupBox("Analysis")
        analyse_area = QGridLayout()
        analyse_group.setLayout(analyse_area)
        grid_group = QGroupBox("Grid")
        grid_area = QGridLayout()
        grid_group.setLayout(grid_area)
        result_group = QGroupBox("Result")
        result_area = QGridLayout()
        result_group.setLayout(result_area)

        self.tabs = QTabWidget()

        # Create a plot widget for the audio clip spectrum and add it to the
        # tab
        self.clipSpectrumPlot = pg.PlotWidget()
        self.clipSpectrumPlot.setLabel("left", "Amplitude")
        self.clipSpectrumPlot.setLabel("bottom", "Frequency (Hz)")
        self.clipSpectrumPlot.addLegend()
        self.clipSpectrumPlot.setBackground("w")
        self.clipSpectrumPlot.showGrid(x=True, y=True)
        self.clipSpectrumPlot.setXRange(0, 1000)
        self.clipSpectrumPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        self.clipSpectrumCurve = self.clipSpectrumPlot.plot(name="WAV file spectrum",
                                           pen=pg.mkPen(color=(255, 0, 0)))
        self.tabs.addTab(self.clipSpectrumPlot, "Clip Spectrum")

        # Create a plot widget for the various ENF curves and add it to the
        # tab
        self.enfPlot = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.enfPlot.setLabel("left", "Frequency (Hz)")
        self.enfPlot.setLabel("bottom", "Date and time")
        self.enfPlot.addLegend()
        self.enfPlot.setBackground("w")
        self.enfPlot.showGrid(x=True, y=True)
        self.enfPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        self.enfAudioCurve = self.enfPlot.plot(name="ENF values of WAV file",
                                               pen=pg.mkPen(color=(255, 128, 0)))
        self.enfAudioCurveSmothed = self.enfPlot.plot(name="Smoothed ENF values of WAV file",
                                               pen=pg.mkPen(color=(204, 0, 0)))
        self.enfGridCurve = self.enfPlot.plot(name="Grid frequency history",
                                               pen=pg.mkPen(color=(0, 102, 102)))
        self.tabs.addTab(self.enfPlot, "ENF Series")

        # Plots the correlation versus time offset
        self.correlationPlot = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.correlationPlot.setLabel("left", "correlation")
        self.correlationPlot.setLabel("bottom", "Date / time")
        self.correlationPlot.addLegend()
        self.correlationPlot.setBackground("w")
        self.correlationPlot.showGrid(x=True, y=True)
        self.correlationPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        self.correlationCurve = self.correlationPlot.plot(name="Correlation",
                                                   pen=pg.mkPen(color=(255, 0, 255)))
        self.tabs.addTab(self.correlationPlot, "Correlation")

        main_layout.addWidget(self.tabs)

        # Overall layout
        main_layout.addWidget(audio_group)
        main_layout.addWidget(analyse_group)
        main_layout.addWidget(grid_group)
        main_layout.addWidget(result_group)

        # 'audio' area
        self.b_load = QPushButton("Load")
        self.b_load.setToolTip("Load a WAV file to analyse.")
        self.b_load.clicked.connect(self._onOpenFileClicked)
        audio_area.addWidget(self.b_load, 0, 0)
        self.e_fileName = QLineEdit()
        self.e_fileName.setReadOnly(True)
        self.e_fileName.setToolTip("WAV file that has been loaded.")
        audio_area.addWidget(self.e_fileName, 0, 1, 1, 3)
        audio_area.addWidget(QLabel("Sample rate (Hz)"), 1, 0)
        self.e_sampleRate = QLineEdit()
        self.e_sampleRate.setReadOnly(True)
        audio_area.addWidget(self.e_sampleRate, 1, 1)
        audio_area.addWidget(QLabel("Duration (sec)"), 1, 2)
        self.e_duration = QLineEdit()
        self.e_duration.setReadOnly(True)
        audio_area.addWidget(self.e_duration, 1, 3)
        audio_area.setColumnStretch(5, 1)

        # 'Analyse' area; contains settings to get the ENF values from the
        # recorded audio clip
        analyse_area.addWidget(QLabel("Nominal grid freq"), 0, 1)
        self.b_nominal_freq = QComboBox()
        self.b_nominal_freq.addItems(("50", "60"))
        self.b_nominal_freq.setToolTip("The nominal frequency of the power grid at the place of the recording;"
                                       " 50 Hz in most countries.")
        analyse_area.addWidget(self.b_nominal_freq, 0, 2)
        analyse_area.addWidget(QLabel("Band width"), 0, 3)
        self.b_band_size = QSpinBox()
        self.b_band_size.setRange(0, 500)
        self.b_band_size.setValue(200)
        self.b_band_size.setMinimumWidth(100)
        self.b_band_size.setSuffix(" mHz")
        analyse_area.addWidget(self.b_band_size, 0, 4)
        analyse_area.addWidget(QLabel("Harmonic"), 0, 5)
        self.b_harmonic = QSpinBox()
        self.b_harmonic.setRange(1, 10)
        self.b_harmonic.setValue(2)
        analyse_area.addWidget(self.b_harmonic, 0, 6)
        self.c_rem_outliers = QCheckBox("Remove outliers")
        analyse_area.addWidget(self.c_rem_outliers, 1, 0)
        analyse_area.addWidget(QLabel("Threshold"), 1, 1)
        self.sp_Outlier_Threshold = QDoubleSpinBox(self)
        self.sp_Outlier_Threshold.setValue(3)
        self.sp_Outlier_Threshold.setToolTip("Factor defining which ENF values shall be considered invalid outliers")
        analyse_area.addWidget(self.sp_Outlier_Threshold,1, 2)
        analyse_area.addWidget(QLabel("Window"), 1, 3)
        self.sp_window = QSpinBox()
        self.sp_window.setValue(5)
        analyse_area.addWidget(self.sp_window,1, 4)

        analyse_area.setColumnStretch(6, 1)

        self.b_analyse = QPushButton("Analyse")
        self.b_analyse.clicked.connect(self.__onAnalyseClicked)
        analyse_area.addWidget(self.b_analyse, 2, 0)

        # 'Grid' area; settings to download the ENF values from the internet
        grid_area.addWidget(QLabel("Location"), 0, 0)
        self.l_country = QComboBox(self)
        for l in GridDataAccessFactory.enumLocations():
            self.l_country.addItem(l)
        self.l_country.addItem("Test")
        grid_area.addWidget(self.l_country, 0, 1)
        grid_area.addWidget(QLabel("Year"), 0, 2)
        self.l_year = QComboBox(self)
        for y in range(2024, 2000 - 1, -1):
            self.l_year.addItem(f'{y}')
        grid_area.addWidget(self.l_year, 0, 3)
        grid_area.addWidget(QLabel("Month"), 0, 4)
        self.l_month = QComboBox()
        self.l_month.addItems(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        grid_area.addWidget(self.l_month, 0, 5)
        self.b_loadGridHistory = QPushButton("Load")
        grid_area.addWidget(self.b_loadGridHistory, 1, 0)
        self.b_loadGridHistory.clicked.connect(self.__onLoadGridHistoryClicked)
        grid_area.setColumnStretch(6, 1)


        self.b_match = QPushButton("Match")
        self.b_match.clicked.connect(self.__onMatchClicked)
        result_area.addWidget(self.b_match, 0, 0)
        self.cb_algo = QComboBox()
        self.cb_algo.addItems(('Convolution', 'Euclidian', 'Pearson'))
        result_area.addWidget(self.cb_algo, 0, 1)
        result_area.addWidget(QLabel("Offset (sec)"), 1, 0)
        self.e_offset = QLineEdit()
        self.e_offset.setReadOnly(True)
        result_area.addWidget(self.e_offset, 1, 1)
        result_area.addWidget(QLabel("Date / time"), 1, 2)
        self.e_date = QLineEdit()
        self.e_date.setReadOnly(True)
        result_area.addWidget(self.e_date, 1, 3)
        result_area.addWidget(QLabel("Quality"), 1, 4)
        self.e_quality = QLineEdit()
        self.e_quality.setReadOnly(True)
        result_area.addWidget(self.e_quality, 1, 5)

        self.__setButtonStatus()

        widget.setLayout(main_layout)
        self.setCentralWidget(widget)


    def __createMenu(self):
        """Create a menu."""
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)

        file_menu = menuBar.addMenu("&File")

        b_open = QAction("&Open project", self)
        b_open.setStatusTip("Open a project")
        b_save = QAction("&Save project", self)
        b_save.setStatusTip("Save the project")
        showEnfSourcesAction = QAction("Show &ENF sources", self)
        showEnfSourcesAction.triggered.connect(self.__showEnfSources)

        file_menu.addAction(showEnfSourcesAction)
        file_menu.addAction(b_open)
        file_menu.addAction(b_save)

        editMenu = menuBar.addMenu("&Edit")

        editSettingsAction = QAction("&Settings", self)
        editSettingsAction.triggered.connect(self.__editSettings)
        editMenu.addAction(editSettingsAction)


    def __editSettings(self):
        """Menu item; pops up a 'setting' dialog."""
        # TODO: Sort out settings. Probbly call 'newSettings' method of all
        # embbeded objects.
        dlg = SettingsDialog(self.settings)
        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")


    def __showEnfSources(self):
        self.setCursor(Qt.WaitCursor)
        dlg = ShowEnfSourcesDlg(self)
        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")
        self.unsetCursor()


    def __setButtonStatus(self):
        """ Enables or disables button depending on the clip status."""
        audioDataLoaded = self.clip is not None and self.clip.getData() is not None
        audioEnfLoaded = self.clip is not None and self.clip.getENF() is not None
        gridEnfLoaded = self.grid is not None and self.grid.getENF() is not None

        self.b_analyse.setEnabled(audioDataLoaded)
        self.b_match.setEnabled(audioEnfLoaded and gridEnfLoaded)


    def __plotAudioRec(self, timestamp=0):
        """ Plot the ENF values of an audio recording.

        :param timestamp: The timestamp of the clip. Should be specified when the clip
        has been matched against the grid frequencies.
        :param clip.getFFT():
        :param clip.getENF():
        :param clip.getENFs():

        | Clip ENF | Grid ENF | Matched | Displayed curves  | Time scale           |
        |----------+----------+---------+-------------------+----------------------|
        | yes      |          |         | Clip ENF          | Starting at 0        |
        | yes      | yes      |         | Clip und Grid ENF | Starting at Grid ENF |
        |          | yes      |         | Grid ENF          | Starting at Grid ENF |
        | yes      | yes      | yes     | Clip und Grid ENF | Match posiiton       |

        """
        # FIXME: Sinnlos, wenn der Clip noch nicht analysiert ist.
        assert(type(self.clip) == EnfModel)

        # Plot FFT of the clip (if already computed)
        fft_t, fft_a = self.clip.getFFT()
        if fft_t is not None and fft_a is not None:
            self.clipSpectrumCurve.setData(fft_t, fft_a)

        # Plot ENF of clip (if already computed)
        data = self.clip.getENF()
        if data is not None:
            self.enfAudioCurve.setData(list(range(timestamp, len(data) + timestamp)),
                                       data)
        smoothedData = self.clip.getENFs()
        if smoothedData is not None:
            self.enfAudioCurveSmothed.setData(list(range(timestamp, len(smoothedData) + timestamp)),
                                       smoothedData)

        self.e_sampleRate.setText(str(self.clip.sampleRate()))


    def __plotGridHistory(self):
        """ Plot the grid frequency history.
        """
        data = self.grid.getENF()
        if data is not None:
            assert(type(data) == np.ndarray)
            timestamp = self.grid.getTimestamp()
            assert timestamp is not None or type(timestamp) == int

            print("Seconds from epoch:", timestamp)

            timestamps = range(timestamp, timestamp + len(data))
            self.enfGridCurve.setData(timestamps, data)
        else:
            self.enfGridCurve.setData([])



    def __plotCorrelation(self, t, corr):
        # FIXME: Old curves/legends are not removed from plot if a new oneis drawn
        self.correlationCurve.setData(list(range(len(corr))), corr)


    def __showMatches(self, t, q, corr):
        """Display the result of matching the clips ENF versus the grid ENF.

        :param t: The timestamp of the match in seconds since the epoch.
        :param q: Descibes the quelity of the match. Interpretation depends
        on the matching algorithm.
        :param corr:

        """
        # print("Show matches")
        duration = self.clip.getDuration()

        self.e_offset.setText(str(t))
        self.e_quality.setText(str(q))
        self.enfPlot.setXRange(t, t + duration, padding=1)
        ts = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
        self.e_date.setText(ts)
        self.__plotCorrelation(t, corr)


    @classmethod
    def convertToWavFile(cls, fn, tmpfn):
        """ Convert a multimedia file to a WAV file.

        :param fn: The input file name
        :param tmp fn: Temporary output file in WAV format.
        """
        cmd = ["/usr/bin/ffmpeg", "-i", fn, "-ar", "4000",  "-ac", "1", "-f",
               "wav", tmpfn]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        output, errors = p.communicate()
        print("Output:", output)
        print("Errors:", errors)
        return p.returncode == 0


    def _onOpenFileClicked(self):
        """ Choose a WAV file woth an audio recording."""
        self.setCursor(Qt.WaitCursor)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, x = QFileDialog.getOpenFileName(self,"Open audio or video file",
                                                  "", "all files (*)",
                                                  options=options)
        if fileName and fileName != '':
            self.clip = EnfModel(self.databasePath)
            tmpfn = f"/tmp/hum-tmp-{os.getpid()}.wav"
            # TODO: Check for errors
            self.convertToWavFile(fileName, tmpfn)
            self.clip.fromWaveFile(tmpfn)
            self.e_fileName.setText(fileName)
            self.e_duration.setText(str(self.clip.getDuration()))
            self.e_sampleRate.setText(str(self.clip.sampleRate()))
            os.remove(tmpfn)

            # Clear all clip-related plots
            self.enfAudioCurve.setData([], [])
            self.enfAudioCurveSmothed.setData([])
            self.correlationCurve.setData([], [])
            self.clipSpectrumCurve.setData([], [])
            self.__plotAudioRec()

        self.unsetCursor()
        self.__setButtonStatus()


    def __onAnalyseClicked(self):
        """ Called when the 'analyse' button is pressed. """
        # Display wait cursor
        self.setCursor(Qt.WaitCursor)

        self.clip.makeEnf(int(self.b_nominal_freq.currentText()),
                           float(self.b_band_size.value()/1000),
                           int(self.b_harmonic.value()))
        if self.c_rem_outliers.isChecked():
            m = self.sp_Outlier_Threshold.value()
            window = self.sp_window.value()
            self.clip.outlierSmoother(m, window)
        self.clip.makeFFT()
        gridtimestamp = self.grid.getTimestamp() if self.grid is not None else 0
        self.__plotAudioRec(gridtimestamp)

        self.unsetCursor()
        self.tabs.setCurrentIndex(1)
        self.__setButtonStatus()


    def __onLoadGridHistoryClicked(self):
        """ Gets historical ENF values from an ENF database. Called when the 'load' button
        in the 'grid' field is clicked."""
        self.setCursor(Qt.WaitCursor)

        # TODO: Clear old grid history plot

        location = self.l_country.currentText()
        year = int(self.l_year.currentText())
        month = self.l_month.currentIndex() + 1
        self.grid = EnfModel(self.settings.databasePath())

        # Clear old curve
        self.enfGridCurve.setData([], [])
        self.__plotGridHistory()

        if location == 'Test':
            self.grid.fromWaveFile("71000_ref.wav")
            self.grid.makeEnf(int(self.b_nominal_freq.currentText()),
                            float(self.b_band_size.value()/1000),
                            int(self.b_harmonic.value()))
        else:
            self.grid.loadGridEnf(location, year, month)
        if self.grid.enf is not None:
            self.__plotGridHistory()
        else:
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Information")
            dlg.setIcon(QMessageBox.Information)
            dlg.setText(f"Could not get {location} ENF data for {year}-{month:02}")
            dlg.exec()
        if self.clip is not None:
            self.__plotAudioRec(timestamp=self.grid.getTimestamp())
        self.unsetCursor()
        self.tabs.setCurrentIndex(1)
        self.__setButtonStatus()


    def __onMatchClicked(self):
        """Called when the 'match' button is clicked.

        The method finds the best match of the ENF series of the clip
        (self.clip) and the ENF series of the chosen grid (self.grid).
        Result of the matching process are the values: (1) The time
        offset in seconds from the beginning of the grid ENF,
        (2) a quality indication, and (3) an array of correlation values.
        """
        # self.setCursor(Qt.WaitCursor)

        now = datetime.datetime.now()
        print(f"{now} ... starting")
        algo = self.cb_algo.currentText()
        assert algo in ('Convolution', 'Pearson', 'Euclidian')

        ## Progress dialog
        matchingSteps = self.clip.getMatchingSteps(self.grid)
        self.progressDialog = QProgressDialog("Trying to locate audio recording, computing best fit ...", "Abort",
                                              0, matchingSteps, self)
        self.progressDialog.setWindowTitle("Matching clip")
        self.progressDialog.setWindowModality(Qt.WindowModal)
        self.progressDialog.canceled.connect(self.clip.onCanceled)

        if algo == 'Pearson':
            t, q, corr = self.clip.matchPearson(self.grid, self.matchingProgress)
        elif algo == 'Euclidian':
            t, q, corr = self.clip.matchEuclidianDist(self.grid, self.matchingProgress)
        elif algo == 'Convolution':
            t, q, corr = self.clip.matchConv(self.grid, self.matchingProgress)
        if corr is not None:
            self.__showMatches(t, q, corr)
            self.__plotAudioRec(timestamp=t)
            # self.controller.__onMatchClicked(self.cb_algo.currentText())
            self.tabs.setCurrentIndex(1)
            # self.unsetCursor()
            now = datetime.datetime.now()
            print(f"{now} ... done")
        self.__setButtonStatus()


    def matchingProgress(self, value):
        """Called by matchXxxx method to indicate the matching progress."""
        self.progressDialog.setValue(value)


class ShowEnfSourcesDlg(QDialog):

    def __init__(self, parent):
        super().__init__(parent)

        columns = ("Grid/country", "From", "To")
        locations = [r for r in GridDataAccessFactory.enumLocations()]

        self.setWindowTitle("ENF Data Sources")

        self.layout = QVBoxLayout()

        self.layout.addWidget(QLabel("Available date range for each grid/country:"))

        self.table_layout = QGridLayout()

        # Column headers
        for i in range(len(columns)):
            h = QLabel(columns[i])
            h.setStyleSheet("font-weight: bold")
            self.table_layout.addWidget(h, 0, i)

        # Rows
        for i in range(len(locations)):
            f = GridDataAccessFactory.getInstance(locations[i], parent.databasePath)
            fromDate, toDate = f.getDateRange()
            self.table_layout.addWidget(QLabel(locations[i]), i+1, 0)
            self.table_layout.addWidget(QLabel(fromDate), i+1, 1)
            #self.table_layout.addWidget(QLineEdit(fromDate), i+1, 1)
            self.table_layout.addWidget(QLabel(toDate), i+1, 2)

        self.layout.addLayout(self.table_layout)

        QBtn = QDialogButtonBox.Ok
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)


class SettingsDialog(QDialog):

    def __init__(self, settings):
        super().__init__()

        assert(type(settings) == Settings)

        self.settings = settings
        self.setWindowTitle("Edit Settings")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.save)
        self.buttonBox.rejected.connect(self.reject)

        top_layout = QVBoxLayout()
        layout = QGridLayout()
        top_layout.addLayout(layout)
        top_layout.addWidget(self.buttonBox)
        layout.addWidget(QLabel("Database path:"), 0, 0)
        self.e_databasePath = QLineEdit()
        layout.addWidget(self.e_databasePath, 0, 1)
        self.e_databasePath.setToolTip("Path where downlaeded ENF data are stored")
        self.setLayout(top_layout)

        self.e_databasePath.setText(self.settings.databasePath())


    def save(self):
        self.settings.setDatabasePath(self.e_databasePath.text())
        self.settings.save()
        self.accept()


class Settings():
    """ Keep track of settings."""

    template = {"databasepath": "/tmp/hum.sqlite"}

    def __init__(self):
        """ Initialise the setting.

        Attempt to read the settings from a JSON file. Its path is hard-coded as '~/.hum.json'.
        If it does not exist or is malformed, default values are used. Internally, the values are
        stored in a dict.
        """
        print("Loading settings ...")

        # File where settings are stored
        self.settingsPath = os.path.expanduser("~") + "/.hum.json"

        try:
            with open(self.settingsPath, 'r') as s:
                self.settings0 = json.load(s)
                print("... OK")
        except IOError:
            print("... Not found")
            self.settings0 = {}
        except Exception as e:
            self.settings0 = {}
            print(e)
        self.__setDefaults()
        self.settings = self.settings0.copy()


    def save(self):
        """ Save the settings to a JSON file.

        The method checks if the settings have actually been modified and
        if so writes them to a file.
        """
        print("Saving settings ...")

        # If values have changed then save the settings
        if self.settings != self.settings0:
            print("... not equeal ...")
            try:
                with open(self.settingsPath, 'w') as s:
                    json_object = json.dumps(self.settings, indent=4)
                    s.write(json_object)
                    self.settings0 = self.settings.copy()
                    print("... OK")
            except IOError as e:
                print("... Exception:", e)
        else:
            print("... not changed")


    def __setDefaults(self):
        for item in Settings.template:
            if not item in self.settings0:
                self.settings0[item] = Settings.template[item]


    def databasePath(self):
        """ Get the database path from the settings."""
        return self.settings["databasepath"]


    def setDatabasePath(self, path):
        self.settings['databasepath'] = path


class HumController(QApplication):
    """ Orchestrate view and clip. """
    def __init__(self, argv):
        super(HumController, self).__init__(argv)
        self.view = HumView()

    def show(self):
        self.view.show()


if __name__ == '__main__':
    try:
        app = HumController([])
        app.show()
        app.exec()
    except MemoryError as e:
        print(e)
