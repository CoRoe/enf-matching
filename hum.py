#!/usr/bin/python3

# https://www.pythonguis.com/tutorials/plotting-pyqtgraph/'


import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
                             QVBoxLayout, QLineEdit, QFileDialog, QLabel,
                             QPushButton, QGroupBox, QGridLayout,
                             QComboBox, QSpinBox, QTabWidget,
                             QMenu, QMenuBar, QAction, QDialog,
                             QDialogButtonBox)
from PyQt5.Qt import Qt, QIcon

from scipy import signal, fft
from numpy import uint16
import wave
import numpy as np
import datetime
import os
import json
import requests
import h5py



def pmcc(x, y):
    """Calculates the PMCC between x and y data points.

    :param x: list of x values
    :param y: list of y values, same length as x
    :returns: PMCC of x and y, as a float
    """
    return np.corrcoef(x, y)[0][1]


def sorted_pmccs(target, references):
    """Calculates and sorts PMCCs between `target` and each of `references`.

    :param target: list of target data points
    :param references: list of lists of reference data points
    :returns: list of tuples of (reference index, PMCC), sorted desc by PMCC
    """
    pmccs = [pmcc(target, r) for r in references]
    sorted_pmccs = [(idx, v) for idx, v in sorted(enumerate(pmccs), key=lambda item: -item[1])]

    return sorted_pmccs


def search(target_enf, reference_enf):
    """Calculates PMCCs between `target_enf` and each window in `reference_enf`.

    :param target_enf: list of target's ENF values
    :param reference_enf: list of reference's ENF values
    :returns: list of tuples of (reference index, PMCC), sorted desc by PMCC
    """
    n_steps = len(reference_enf) - len(target_enf)
    reference_enfs = (reference_enf[step:step+len(target_enf)] for step in range(n_steps))

    coeffs = sorted_pmccs(target_enf, reference_enfs)
    return coeffs


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
            't': t,
            'Zxx': Zxx,
        },
        'enf': [f/float(harmonic_n) for f in max_freqs],
    }


class EnfModel():
    """Models an ENF series.

    Has methods to retrieve ENF series from databases on the internet.
    """

    def __init__(self, databasePath):
        """Create an empty ENF series.

        :param dataBasePath: File system path where the database will be
        cached.
        """
        self.data = None
        self.enf = None
        self.databasePath = databasePath
        #self.fft = None
        self.fft_freq = None
        self.fft_freq = None
        #self.nominal_freq = 50
        #self.freq_band_size = 0.2
        #self.t0 = datetime.datetime(0, 0, 0)


    def setDatabasePath(self, path):
        self.databasePath = path


    def fromWaveFile(self, fpath):
        """Loads a .wav file and computes ENF and SFT.

        :param fpath: the path to __load the file from rate)

        On exit, self.data is an Numpy array with the samples of the loaded audio recoding.
        """
        with wave.open(fpath) as wav_f:
            wav_buf = wav_f.readframes(wav_f.getnframes())
            self.data = np.frombuffer(wav_buf, dtype=np.int16)
            assert(type(self.data) == np.ndarray)
            self.fs = wav_f.getframerate()
            self.clip_len_s = len(self.data) / self.fs
            print(f"File {fpath}: Sample frequency {self.fs} Hz, duration {self.clip_len_s} seconds")


    def makeEnf(self, nominal_freq, freq_band_size, harmonic):
        """ Creates an ENF series from the sample data.

        :param: nominal_freq: Nominal grid frequency in Hz; usually 50 or 60 Hz
        :param: freq_band_size: Size of the frequency band around *nominal_freq* in Hz
        :param: harmonic:
        """
        assert(self.data is not None)

        self.nominal_freq = nominal_freq
        self.freq_band_size = freq_band_size
        self.harmonic = harmonic
        self.enf_output = enf_series(self.data, self.fs, nominal_freq,
                                     freq_band_size, harmonic_n=harmonic)

        # stft is the Short-Term Fourier Transfrom of the autio file, computed
        # per second.
        self.stft = self.enf_output['stft']

        # ENF are the ENF values
        # self.enf = self.enf_output['enf']
        enf = [int(e * 1000) for e in self.enf_output['enf']]
        self.enf = np.array(enf)
        # print(self.enf[0:5])


    def loadGridEnf(self, location, year: int, month: int):
        """ Load the grid ENF values from a database.

        :param location: The name/location of the grid
        :param year: The year
        :param month: The number of the month (1 = Jan, 2 = Feb, ...)
        """
        assert(self.databasePath)
        assert(type(year) == int and year > 1970)
        assert(type(month) == int and month >= 1 and month <= 12)

        fileName = f"{self.databasePath}/{location}.hp5"
        timestamp = f"{year}-{month}-01 00:00:00"

        # First check if the enf series is contained in the database
        try:
            with h5py.File(fileName, 'r') as f:
                dset = f[timestamp]
                self.enf = dset[()]
                print(f"From database: {type(self.enf)}")
        except:
            if location == 'GB':
                enf = self.loadNationalGridGB(location, year, month)
                if enf is not None:
                    assert(type(enf) == np.ndarray)
                    print(f"National grid date is type {type(enf)}")
                    self.enf = enf
                    try:
                        with h5py.File(fileName, 'w') as f:
                            dset = f.create_dataset(timestamp, data=enf)
                    except Exception as e:
                        print("Failed to write enf to datebase:", e)
        print()


    def loadNationalGridGB(self, location, year, month):
        """
        Download ENF historical data from the GB National database.

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
                    data = [uint16(float(row.split(',')[1]) * 1000) for row in
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


    def match(self, ref):
        """Given a reference model with ENF values find the best fit with the
        own ENF values.

        :param ref: The ENF series of the grid
        :returns: The index into the reference series of thebets match; the
        correlation at that index, an array of correlations for all possible
        indices.

        The method computes the Pearson correlation between the ENF values in
        the clip and the grid.

        See: https://realpython.com/numpy-scipy-pandas-correlation-python/
        https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        """
        assert(type(ref) == EnfModel)

        print(f"Start correlation computation: {datetime.datetime.now()} ...")
        ref_enf = ref.getENF()
        n_steps = len(ref_enf) - len(self.enf) + 1
        corr = [np.corrcoef(ref_enf[step:step+len(self.enf)], self.enf)[0][1]
                for step in range(n_steps)]
        max_index = np.argmax(corr)
        print(f"End correlation computation {datetime.datetime.now()} ...")
        return max_index, corr[max_index], corr


    def getENF(self):
        return self.enf


    def getData(self):
        return self.data


    def getFFT(self):
        return self.fft_freq, self.fft_ampl


    def duration(self):
        """ Length of the clip in seconds."""
        assert(self.enf is not None)
        return self.clip_len_s


    def sampleRate(self):
        """ Return the sample rate in samples / second. """
        return self.fs


class HumView(QMainWindow):
    """ Display ENF analysis and result."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.audioCurve = None
        self.gridFreqCurve = None
        self.correlationCurve = None
        self.createWidgets()
        self.createMenu()


    def createWidgets(self):
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

        self.fftPlot = pg.PlotWidget()
        self.fftPlot.setLabel("left", "Amplitude")
        self.fftPlot.setLabel("bottom", "Frequency (Hz)")
        self.fftPlot.addLegend()
        self.fftPlot.setBackground("w")
        self.fftPlot.showGrid(x=True, y=True)
        self.fftPlot.setXRange(0, 1000)
        self.fftPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        self.tabs.addTab(self.fftPlot, "FFT")

        # Widget showing the ENF values of a grid and an audio recording
        #
        # See https://pyqtgraph.readthedocs.io/en/latest/getting_started/plotting.html
        self.enfPlot = pg.PlotWidget()
        self.enfPlot.setLabel("left", "Frequency (Hz)")
        self.enfPlot.setLabel("bottom", "Time (sec)")
        self.enfPlot.addLegend()
        self.enfPlot.setBackground("w")
        self.enfPlot.showGrid(x=True, y=True)
        self.enfPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        self.tabs.addTab(self.enfPlot, "ENF")

        # Plots the correlation versus time offset
        self.corrPlot = pg.PlotWidget()
        self.corrPlot.setLabel("left", "correlation")
        self.corrPlot.setLabel("bottom", "time lag [sec]")
        self.corrPlot.addLegend()
        self.corrPlot.setBackground("w")
        self.corrPlot.showGrid(x=True, y=True)
        self.corrPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        #self.corrPlot.setYRange(-1, +1)
        self.tabs.addTab(self.corrPlot, "Correlation")

        main_layout.addWidget(self.tabs)

        main_layout.addLayout(grid_area)
        main_layout.addWidget(audio_group)
        main_layout.addWidget(analyse_group)
        main_layout.addWidget(grid_group)
        main_layout.addWidget(result_group)

        self.b_load = QPushButton("Load")
        self.b_load.setToolTip("Load a WAV file to analyse.")
        self.b_load.clicked.connect(self.onOpenWavFile)
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

        grid_area.addWidget(QLabel("Location"), 0, 0)
        self.l_country = QComboBox(self)
        self.l_country.addItems(("Test", "GB"))
        grid_area.addWidget(self.l_country, 0, 1)
        grid_area.addWidget(QLabel("Year"), 0, 2)
        self.l_year = QComboBox(self)
        for y in range(2000, 2023 + 1):
            self.l_year.addItem(f'{y}')
        grid_area.addWidget(self.l_year, 0, 3)
        grid_area.addWidget(QLabel("Month"), 0, 4)
        self.l_month = QComboBox()
        self.l_month.addItems(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        grid_area.addWidget(self.l_month, 0, 5)
        self.b_loadGridHistory = QPushButton("Load")
        grid_area.addWidget(self.b_loadGridHistory, 1, 0)
        self.b_loadGridHistory.clicked.connect(self.onLoadGridHistory)
        grid_area.setColumnStretch(6, 1)

        analyse_area.addWidget(QLabel("Nominal grid freq"), 0, 0)
        self.b_nominal_freq = QComboBox()
        self.b_nominal_freq.addItems(("50", "60"))
        self.b_nominal_freq.setToolTip("The nominal frequency of the power grid at the place of the recording;"
                                       " 50 Hz in most countries.")
        analyse_area.addWidget(self.b_nominal_freq, 0, 1)
        analyse_area.addWidget(QLabel("Band width"), 0, 2)
        self.b_band_size = QSpinBox()
        self.b_band_size.setRange(0, 500)
        self.b_band_size.setValue(200)
        self.b_band_size.setMinimumWidth(100)
        self.b_band_size.setSuffix(" mHz")
        analyse_area.addWidget(self.b_band_size, 0, 3)
        analyse_area.addWidget(QLabel("Harmonic"), 0, 4)
        self.b_harmonic = QSpinBox()
        self.b_harmonic.setRange(1, 10)
        self.b_harmonic.setValue(2)
        analyse_area.addWidget(self.b_harmonic, 0, 5)
        analyse_area.setColumnStretch(6, 1)

        self.b_analyse = QPushButton("Analyse")
        self.b_analyse.clicked.connect(self.onAnalyse)
        analyse_area.addWidget(self.b_analyse, 1, 0)

        self.b_match = QPushButton("Match")
        self.b_match.clicked.connect(self.onMatch)
        result_area.addWidget(self.b_match, 0, 0)
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

        self.setButtonStatus()

        widget.setLayout(main_layout)
        self.setCentralWidget(widget)


    def createMenu(self):
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)

        b_open = QAction("&Open project", self)
        b_open.setStatusTip("Open a project")
        b_save = QAction("&Save project", self)
        b_save.setStatusTip("Save the project")

        file_menu = menuBar.addMenu("&File")
        file_menu.addAction(b_open)
        file_menu.addAction(b_save)

        editMenu = menuBar.addMenu("&Edit")

        editSettingsAction = QAction("&Settings", self)
        editSettingsAction.triggered.connect(self.editSettings)
        editMenu.addAction(editSettingsAction)


    def editSettings(self):
        dlg = SettingsDialog(self.controller.settings)
        if dlg.exec():
            print("Success!")
            #dlg.save()
            # self.controller.setDatabasePath(dlg.databasePath())
        else:
            print("Cancel!")


    def setButtonStatus(self):
        """ Enables or disables button depending on the model status."""
        audioDataLoaded = self.controller.audioData() is not None
        audioEnfLoaded = self.controller.audioENF() is not None
        gridEnfLoaded = self.controller.gridENF() is not None

        self.b_analyse.setEnabled(audioDataLoaded)
        self.b_match.setEnabled(audioEnfLoaded and gridEnfLoaded)


    def plotAudioRec(self, audioRecording, t_offset=0):
        """ Plot the ENF values of an audio recording."""
        assert(type(audioRecording) == EnfModel)

        data = audioRecording.getENF()
        pen = pg.mkPen(color=(255, 0, 0))

        fft_t, fft_a = audioRecording.makeFFT()
        self.fft_curve = self.fftPlot.plot(name="WAV file spectrum",
                                           pen=pen)
        self.fft_curve.setData(fft_t, fft_a)

        if self.audioCurve:
            self.enfPlot.removeItem(self.audioCurve)
        self.audioCurve = self.enfPlot.plot(name="ENF values of WAV file",
                                               pen=pen)
        self.audioCurve.setData(list(range(t_offset, len(data) + t_offset)),
                                data)

        self.e_duration.setText(str(audioRecording.duration()))
        self.e_sampleRate.setText(str(audioRecording.sampleRate()))


    def plotGridHistory(self, gridHistory):
        """ Plot the grid frequency history.

        :param: gridHistory - instanceof the grid history model
        """
        assert(type(gridHistory == EnfModel))

        data = gridHistory.getENF()
        assert(data is not None and type(data) == np.ndarray)

        pen = pg.mkPen(color=(0, 0, 255))

        if self.gridFreqCurve:
            self.enfPlot.removeItem(self.gridFreqCurve)
        self.gridFreqCurve = self.enfPlot.plot(name="Grid frequency history", pen=pen)
        self.gridFreqCurve.setData(list(range(len(data))), data)


    def plotCorrelation(self, t, corr):
        pen = pg.mkPen(color=(0, 0, 255))

        if self.correlationCurve:
            self.enfPlot.removeItem(self.correlationCurve)
        self.correlationCurve = self.corrPlot.plot(name="Correlation", pen=pen)
        self.correlationCurve.setData(list(range(len(corr))), corr)


    def showMatches(self, t, q, corr):
        # print("Show matches")
        duration = self.controller.getDuration()

        self.e_offset.setText(str(t))
        self.e_quality.setText(str(q))
        self.enfPlot.setXRange(t, t + duration, padding=1)

        self.plotCorrelation(t, corr)


    def onOpenWavFile(self):
        """ Choose a WAV file woth an audio recording."""
        self.setCursor(Qt.WaitCursor)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, x = QFileDialog.getOpenFileName(self,"Open audio file",
                                                  "","WAV Files (*.wav);;all files (*)",
                                                  options=options)
        if fileName and fileName != '':
            self.controller.loadAudioFile(fileName)
            self.e_fileName.setText(fileName)
            self.e_duration.setText(str(self.controller.getDuration()))
            self.e_sampleRate.setText(str(self.controller.getSampleRate()))

        self.unsetCursor()
        self.setButtonStatus()


    def onLoadGridHistory(self):
        """ Gets historical ENF values from an ENF database. Called when the 'load' button
        in the 'grid' field is clicked."""
        self.setCursor(Qt.WaitCursor)

        location = self.l_country.currentText()
        year = int(self.l_year.currentText())
        month = self.l_month.currentIndex() + 1
        self.controller.onLoadGridHistory(location, year, month,
                                          int(self.b_nominal_freq.currentText()),
                                float(self.b_band_size.value()/1000),
                                int(self.b_harmonic.value()))
        #self.gridHistory = GridHistoryModel(location, year, month)

        self.unsetCursor()
        self.tabs.setCurrentIndex(1)
        self.setButtonStatus()


    def onAnalyse(self):
        """ Called when the 'analyse' button is pressed. """
        self.setCursor(Qt.WaitCursor)

        #harmonic = int(self.b_nominal_freq, self.b_band_size, self.b_harmonic.value())
        if self.audioCurve:
            # FIXME: Curve still visible
            self.audioCurve.clear()
            #self.enfPlot.removeItem(self.audioCurve)
            #self.audioCurve = None
        self.controller.onAnalyse(int(self.b_nominal_freq.currentText()),
                                float(self.b_band_size.value()/1000),
                                int(self.b_harmonic.value()))

        self.unsetCursor()
        self.setButtonStatus()


    def onMatch(self):
        """Called when the 'match' button is clicked."""
        self.setCursor(Qt.WaitCursor)
        now = datetime.datetime.now()
        print(f"{now} ... starting")
        self.controller.onMatch()
        self.tabs.setCurrentIndex(1)
        self.unsetCursor()
        now = datetime.datetime.now()
        print(f"{now} ... done")
        self.setButtonStatus()


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
        layout.addWidget(QLabel("Database directory:"), 0, 0)
        self.e_databasePath = QLineEdit()
        layout.addWidget(self.e_databasePath, 0, 1)
        self.e_databasePath.setToolTip("Path where downlaeded ENF data are stored")
        self.setLayout(top_layout)

        #self.__load()
        #self.__setDefaults()
        #self.settings = self.settings0

        self.e_databasePath.setText(self.settings.databasePath())

    def save(self):
        self.settings.setDatabasePath(self.e_databasePath.text())
        self.settings.save()
        self.accept()


class Settings():
    """ Keep track of settings."""

    template = {"databasepath": "/tmp"}

    def __init__(self):
        """ Initialise the setting.

        Attempt to read the settings from a JSON file. Its path is hard-coded as '~/.hum.json'.
        If it does not exist or is malformed, default values are used. Internally, the values are
        stored in a dict.
        """
        print("Load settings ...")

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
    """ Orchestrate view and model. """
    def __init__(self, argv):
        super(HumController, self).__init__(argv)
        self.model = None
        self.gm = None
        self.settings = Settings()
        self.view = HumView(self)

    def show(self):
        self.view.show()

    def loadAudioFile(self, fileName):
        """ Create a model from an audio recoding and tell the view to show
        it."""
        self.model = EnfModel(self.settings.databasePath())
        self.model.fromWaveFile(fileName)

    def onLoadGridHistory(self, location, year, month, nominal_freq,
                          freq_band_size, harmonic):
        assert(type(year) == int and year > 1970)
        assert(type(month) == int and month >= 1 and month <= 12)
        assert(type(nominal_freq) == int and nominal_freq in (50, 60))

        self.gm = EnfModel(self.settings.databasePath())
        if location == 'Test':
            self.gm.fromWaveFile("71000_ref.wav")
            self.gm.makeEnf(nominal_freq, freq_band_size, harmonic)
        elif location == 'GB':
            self.gm.loadGridEnf(location, year, month)
        if self.gm.enf is not None:
            self.view.plotGridHistory(self.gm)

    def onAnalyse(self, nominal_freq, freq_band_size, harmonic):
        # TODO: Sollte nur das Audio-Model analysieren
        if self.model is not None:
            self.model.makeEnf(nominal_freq, freq_band_size, harmonic)
            #m = self.model.match(self.gm)
            #self.view.showMatches(m)
            self.view.plotAudioRec(self.model)

    def onMatch(self):
        # TODO: onAnalyse() und onMatch() auseinandersortieren.
        if self.model and self.gm:
            t, q, corr = self.model.match(self.gm)
            self.view.showMatches(t, q, corr)
            self.view.plotAudioRec(self.model, t_offset=t)

    def getDuration(self):
        return self.model.clip_len_s

    def getSampleRate(self):
        return self.model.fs

    def audioData(self):
        return self.model.getData() if self.model is not None else None

    def audioENF(self):
        return self.model.getENF() if self.model is not None else None

    def gridENF(self):
        return self.gm.getENF() if self.gm is not None else None

    def setDatabasePath(self, path):
        self.model.setDatabasePath(path)
        self.gm.setDatabasePath(path)

    def getSettings(self):
        print()

    def updateSettings(self):
        print()


#
# Main
#
if __name__ == '__main__':
    app = HumController([])
    app.show()
    app.exec()
