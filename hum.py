#!/usr/bin/python3

# https://www.pythonguis.com/tutorials/plotting-pyqtgraph/'


import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
                             QVBoxLayout, QLineEdit, QFileDialog, QLabel,
                             QPushButton, QGroupBox, QGridLayout,
                             QComboBox, QSpinBox, QTabWidget,
                             QMenuBar, QAction, QDialog,
                             QDialogButtonBox)
from PyQt5.Qt import Qt

from scipy import signal, fft, spatial
import wave
import numpy as np
import datetime
import os
import json
from griddata import GridDataAccessFactory
from llvmlite.llvmpy import passes


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
        self.fft_freq = None


    def setDatabasePath(self, path):
        self.databasePath = path


    def fromWaveFile(self, fpath):
        """Loads wave_buf .wav file and computes ENF and SFT.

        :param fpath: the path to __load the file from rate)

        On exit, self.data is an Numpy array with the samples of the loaded audio recording ('clip').
        If the WAV file has a sampling rate above 8,000 Hz it is decimated down to 8,000 Hz. The
        function throws an exception if the original sampling rate is not a multiple of 8,000.
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
        assert location != 'Test', "Handled elsewhere"
        data_source = GridDataAccessFactory.getInstance(location, self.databasePath)
        self.enf = data_source.getEnfSeries(year, month)


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


    def matchPearson(self, ref):
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

        print(f"Start Pearson correlation computation: {datetime.datetime.now()} ...")
        ref_enf = ref.getENF()
        n_steps = len(ref_enf) - len(self.enf) + 1
        corr = [np.corrcoef(ref_enf[step:step+len(self.enf)], self.enf)[0][1]
                for step in range(n_steps)]
        max_index = np.argmax(corr)
        print(f"End Pearson correlation computation {datetime.datetime.now()} ...")
        return max_index, corr[max_index], corr


    def matchEuclidianDist(self, ref):
        """Given a reference model with ENF values find the best fit with the
        own ENF values.

        :param ref: The ENF series of the grid
        :returns: The index into the reference series of thebets match; the
        correlation at that index, an array of correlations for all possible
        indices.

        The method computes the Euclidian distance between the ENF values in
        the clip and the grid.

        See: https://www.geeksforgeeks.org/python-distance-between-collections-of-inputs/
        """
        assert(type(ref) == EnfModel)

        print(f"Start Euclidian correlation computation: {datetime.datetime.now()} ...")
        ref_enf = ref.getENF()
        n_steps = len(ref_enf) - len(self.enf) + 1
        corr = [spatial.distance.cdist([ref_enf[step:step+len(self.enf)], self.enf],
                                       [ref_enf[step:step+len(self.enf)], self.enf],
                                       'sqeuclidean')[0][1] for step in range(n_steps)]
        min_index = np.argmin(corr)
        print(f"End Euclidian correlation computation {datetime.datetime.now()} ...")
        return min_index, corr[min_index], corr


    def getENF(self):
        return self.enf


    def getData(self):
        return self.data


    def getFFT(self):
        return self.fft_freq, self.fft_ampl


    def getDuration(self):
        """ Length of the clip in seconds."""
        #assert(self.enf is not None)
        return self.clip_len_s


    def sampleRate(self):
        """ Return the sample rate in samples / second. """
        return self.fs


class HumView(QMainWindow):
    """ Display ENF analysis and result."""

    def __init__(self):
        """Initialize variables and create widgets and menu."""
        super().__init__()
        self.model = None
        self.gm = None
        self.settings = Settings()
        self.databasePath = self.settings.databasePath()

        self.enfAudioCurve = None     # ENF series of loaded audio file
        self.clipSpectrumCurve = None # Fourier transform of loaded audio file
        self.endGridCurve = None      # ENF series of grid
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
        self.enfAudioCurve = self.enfPlot.plot(name="ENF values of WAV file",
                                               pen=pg.mkPen(color=(255, 0, 0)))
        self.endGridCurve = self.enfPlot.plot(name="Grid frequency history",
                                               pen=pg.mkPen(color=(0, 255, 0)))
        self.tabs.addTab(self.enfPlot, "ENF Series")

        # Plots the correlation versus time offset
        self.correlationPlot = pg.PlotWidget()
        self.correlationPlot.setLabel("left", "correlation")
        self.correlationPlot.setLabel("bottom", "time lag [sec]")
        self.correlationPlot.addLegend()
        self.correlationPlot.setBackground("w")
        self.correlationPlot.showGrid(x=True, y=True)
        self.correlationPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        self.correlationCurve = self.correlationPlot.plot(name="Correlation",
                                                   pen=pg.mkPen(color=(255, 0, 255)))
        self.tabs.addTab(self.correlationPlot, "Correlation")

        main_layout.addWidget(self.tabs)

        main_layout.addWidget(audio_group)
        main_layout.addWidget(analyse_group)
        main_layout.addWidget(grid_group)
        main_layout.addWidget(result_group)

        self.b_load = QPushButton("Load")
        self.b_load.setToolTip("Load a WAV file to analyse.")
        self.b_load.clicked.connect(self._onOpenWavFileClicked)
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
        for l in GridDataAccessFactory.enumLocations():
            self.l_country.addItem(l)
        self.l_country.addItem("Test")
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
        self.b_loadGridHistory.clicked.connect(self.__onLoadGridHistoryClicked)
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
        self.b_analyse.clicked.connect(self.__onAnalyseClicked)
        analyse_area.addWidget(self.b_analyse, 1, 0)

        self.b_match = QPushButton("Match")
        self.b_match.clicked.connect(self.__onMatchClicked)
        result_area.addWidget(self.b_match, 0, 0)
        self.cb_algo = QComboBox()
        self.cb_algo.addItems(('Pearson', 'Euclidian'))
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
        """ Enables or disables button depending on the model status."""
        audioDataLoaded = self.model is not None and self.model.getData() is not None
        audioEnfLoaded = self.model is not None and self.model.getENF() is not None
        gridEnfLoaded = self.gm is not None and self.gm.getENF() is not None

        self.b_analyse.setEnabled(audioDataLoaded)
        self.b_match.setEnabled(audioEnfLoaded and gridEnfLoaded)


    def __plotAudioRec(self, audioRecording, t_offset=0):
        """ Plot the ENF values of an audio recording."""
        assert(type(audioRecording) == EnfModel)

        data = audioRecording.getENF()

        # Versuch -------------------------------------------
        # Define legend and curve only once
        fft_t, fft_a = audioRecording.makeFFT()
        self.clipSpectrumCurve.setData(fft_t, fft_a)
        # ---------------------------------------------------

        self.enfAudioCurve.setData(list(range(t_offset, len(data) + t_offset)),
                                data)

        #self.e_duration.setText(str(audioRecording.duration()))
        self.e_sampleRate.setText(str(audioRecording.sampleRate()))


    def __plotGridHistory(self, gridHistory):
        """ Plot the grid frequency history.

        :param: gridHistory - instanceof the grid history model
        """
        # FIXME: Remove curve if gridhistory is None
        assert(type(gridHistory == EnfModel))

        data = gridHistory.getENF()
        assert(data is not None and type(data) == np.ndarray)

        self.endGridCurve.setData(list(range(len(data))), data)


    def __plotCorrelation(self, t, corr):
        # FIXME: Old curves/legends are not removed from plot if a new oneis drawn
        self.correlationCurve.setData(list(range(len(corr))), corr)


    def __showMatches(self, t, q, corr):
        # print("Show matches")
        duration = self.model.getDuration()

        self.e_offset.setText(str(t))
        self.e_quality.setText(str(q))
        self.enfPlot.setXRange(t, t + duration, padding=1)

        self.__plotCorrelation(t, corr)


    def _onOpenWavFileClicked(self):
        """ Choose a WAV file woth an audio recording."""
        self.setCursor(Qt.WaitCursor)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, x = QFileDialog.getOpenFileName(self,"Open audio file",
                                                  "","WAV Files (*.wav);;all files (*)",
                                                  options=options)
        if fileName and fileName != '':
            self.model = EnfModel(self.databasePath)
            #self.model.loadAudioFile(fileName)
            self.model.fromWaveFile(fileName)
            self.e_fileName.setText(fileName)
            self.e_duration.setText(str(self.model.getDuration()))
            self.e_sampleRate.setText(str(self.model.sampleRate()))

            # Clear all plots
            self.enfAudioCurve.setData([], [])
            self.correlationCurve.setData([], [])
            self.clipSpectrumCurve.setData([], [])
            self.endGridCurve.setData([], [])

        self.unsetCursor()
        self.__setButtonStatus()


    def __onLoadGridHistoryClicked(self):
        """ Gets historical ENF values from an ENF database. Called when the 'load' button
        in the 'grid' field is clicked."""
        self.setCursor(Qt.WaitCursor)

        # TODO: Clear old grid history plot

        location = self.l_country.currentText()
        year = int(self.l_year.currentText())
        month = self.l_month.currentIndex() + 1
        self.gm = EnfModel(self.settings.databasePath())
        if location == 'Test':
            self.gm.fromWaveFile("71000_ref.wav")
            self.gm.makeEnf(int(self.b_nominal_freq.currentText()),
                            float(self.b_band_size.value()/1000),
                            int(self.b_harmonic.value()))
        else:
            self.gm.loadGridEnf(location, year, month)
        if self.gm.enf is not None:
            self.__plotGridHistory(self.gm)
        self.unsetCursor()
        self.tabs.setCurrentIndex(1)
        self.__setButtonStatus()


    def __onAnalyseClicked(self):
        """ Called when the 'analyse' button is pressed. """
        # Display wait cursor
        self.setCursor(Qt.WaitCursor)

        self.model.makeEnf(int(self.b_nominal_freq.currentText()),
                           float(self.b_band_size.value()/1000),
                           int(self.b_harmonic.value()))
        self.__plotAudioRec(self.model)

        self.unsetCursor()
        self.tabs.setCurrentIndex(1)
        self.__setButtonStatus()


    def __onMatchClicked(self):
        """Called when the 'match' button is clicked.
        
        The method finds the best match of the ENF series of the clip
        (self.model) and the ENF series of the chosen grid (self.gm).
        Result of the matching process are the values: (1) The time
        offset in seconds from the beginning of the grid ENF,
        (2) a quality indication, and (3) an array of correlation values.
        """
        self.setCursor(Qt.WaitCursor)
        now = datetime.datetime.now()
        print(f"{now} ... starting")
        algo = self.cb_algo.currentText()
        assert algo in ('Pearson', 'Euclidian')
        if algo == 'Pearson':
            t, q, corr = self.model.matchPearson(self.gm)
        elif algo == 'Euclidian':
            t, q, corr = self.model.matchEuclidianDist(self.gm)
        self.__showMatches(t, q, corr)
        self.__plotAudioRec(self.model, t_offset=t)
        # self.controller.__onMatchClicked(self.cb_algo.currentText())
        self.tabs.setCurrentIndex(1)
        self.unsetCursor()
        now = datetime.datetime.now()
        print(f"{now} ... done")
        self.__setButtonStatus()


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
    """ Orchestrate view and model. """
    def __init__(self, argv):
        super(HumController, self).__init__(argv)
        self.view = HumView()

    def show(self):
        self.view.show()


#
# Main
#
if __name__ == '__main__':
    try:
        app = HumController([])
        app.show()
        app.exec()
    except MemoryError as e:
        print(e)
