#!/usr/bin/python3

# https://www.pythonguis.com/tutorials/plotting-pyqtgraph/'


import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
                             QVBoxLayout, QLineEdit, QFileDialog, QLabel, QHBoxLayout, QPushButton, QGroupBox, QGridLayout,
                      QComboBox)
from PyQt5.Qt import Qt, QCoreApplication, QSpinBox

from scipy import signal
import wave
import numpy as np


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
    """Performs a Short-time Fourier Transform (STFT) on input data.

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
    def __init__(self):
        self.data = None
        self.nominal_freq = 50
        self.freq_band_size = 0.2


    def fromWaveFile(self, fpath):
        """Loads a .wav file and computes ENF and SFT.

        :param fpath: the path to load the file from rate)
        """
        with wave.open(fpath) as wav_f:
            wav_buf = wav_f.readframes(wav_f.getnframes())
            self.data = np.frombuffer(wav_buf, dtype=np.int16)
            self.fs = wav_f.getframerate()

            clip_len_s = len(self.data) / self.fs
            print(f"File {fpath}: Sample freqeuncy {self.fs} Hz, duration {clip_len_s} seconds")


    def makeEnf(self, harmonic=1):
        """ Creates an ENF series from the sample data. """
        assert(self.data is not None)
        self.enf_output = enf_series(self.data, self.fs, self.nominal_freq,
                                     self.freq_band_size, harmonic_n=harmonic)
        self.sft = self.enf_output['stft']
        self.enf = self.enf_output['enf']
        # print(self.enf[0:5])


    def match(self, ref):
        assert(type(ref) == EnfModel)
        pmccs = search(self.enf, ref.enf)
        print("Sample  Match")
        #for i in range(10):
        #    print(pmccs[i][0], pmccs[i][1])
        return pmccs


    def getData(self):
        # enf is a
        return self.enf


    def duration(self):
        """ Length of the clip in seconds."""
        assert(self.enf is not None)
        return len(self.enf)


    def sampleRate(self):
        return self.fs


class HumView(QMainWindow):
    """ Displays ENF analysis and result."""

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.audioCurve = None
        self.gridFreqCurve = None
        self.createWidgets()


    def createWidgets(self):
        widget = QWidget()
        self.setWindowTitle("Hum")

        # Define layouts
        main_layout = QVBoxLayout()
        audio_area = QGridLayout()
        audio_group = QGroupBox("Audio")
        audio_group.setLayout(audio_area)
        grid_group = QGroupBox("Grid")
        grid_area = QGridLayout()
        grid_group.setLayout(grid_area)
        analyse_group = QGroupBox("Analysis")
        analyse_area = QGridLayout()
        analyse_group.setLayout(analyse_area)

        # Widget showing the ENF values of a grid and an audio recording
        #
        # See https://pyqtgraph.readthedocs.io/en/latest/getting_started/plotting.html
        self.plotWidget = pg.PlotWidget()
        self.plotWidget.setTitle("Grid frequency vs Time", color="b", size="10pt")
        self.plotWidget.setLabel("left", "Frequency (Hz)")
        self.plotWidget.setLabel("bottom", "Time (sec)")
        self.plotWidget.addLegend()
        self.plotWidget.setBackground("w")
        self.plotWidget.showGrid(x=True, y=True)
        self.plotWidget.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        main_layout.addWidget(self.plotWidget)

        main_layout.addLayout(grid_area)
        main_layout.addWidget(audio_group)
        main_layout.addWidget(grid_group)
        main_layout.addWidget(analyse_group)

        self.b_load = QPushButton("Load")
        self.b_load.clicked.connect(self.onOpenWavFile)
        audio_area.addWidget(self.b_load, 0, 0)
        self.e_fileName = QLineEdit()
        audio_area.addWidget(self.e_fileName, 0, 1)
        audio_area.addWidget(QLabel("Sample rate (Hz)"), 1, 0)
        self.e_sampleRate = QLineEdit()
        audio_area.addWidget(self.e_sampleRate, 1, 1)
        audio_area.addWidget(QLabel("Duration (sec)"), 2, 0)
        self.e_duration = QLineEdit()
        audio_area.addWidget(self.e_duration, 2, 1)

        grid_area.addWidget(QLabel("Location"), 0, 0)
        self.l_country = QComboBox(self)
        self.l_country.addItems(("Test", "GB"))
        grid_area.addWidget(self.l_country, 0, 1)
        grid_area.addWidget(QLabel("Year"), 1, 0)
        self.l_year = QComboBox(self)
        for y in range(2000, 2023 + 1):
            self.l_year.addItem(f'{y}')
        grid_area.addWidget(self.l_year, 1, 1)
        grid_area.addWidget(QLabel("Month"), 1, 2)
        self.l_month = QComboBox()
        self.l_month.addItems(('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'))
        grid_area.addWidget(self.l_month, 1, 3)
        self.b_loadGridHistory = QPushButton("Load")
        grid_area.addWidget(self.b_loadGridHistory, 2, 0)
        self.b_loadGridHistory.clicked.connect(self.onLoadGridHistory)

        self.b_analyse = QPushButton("Analyse")
        self.b_analyse.clicked.connect(self.onAnalyse)
        analyse_area.addWidget(self.b_analyse, 0, 0)
        analyse_area.addWidget(QLabel("Harmonic"), 0, 1)
        self.b_harmonic = QSpinBox()
        self.b_harmonic.setRange(1, 10)
        analyse_area.addWidget(self.b_harmonic, 0, 2)

        analyse_area.addWidget(QLabel("Result"), 1, 0)
        self.e_result = QLineEdit()
        analyse_area.addWidget(self.e_result, 1, 1)

        widget.setLayout(main_layout)
        self.setCentralWidget(widget)


    def plotAudioRec(self, audioRecording, t_offset=0):
        """ Plot the ENF values of an audio recording."""
        assert(type(audioRecording) == EnfModel)

        data = audioRecording.getData()
        pen = pg.mkPen(color=(255, 0, 0))

        if self.audioCurve:
            self.plotWidget.removeItem(self.audioCurve)
        self.audioCurve = self.plotWidget.plot(name="50 Hz spektrum", pen=pen)
        self.audioCurve.setData(list(range(t_offset, len(data) + t_offset)), data)

        self.e_duration.setText(str(audioRecording.duration()))
        self.e_sampleRate.setText(str(audioRecording.sampleRate()))


    def plotGridHistory(self, gridHistory):
        """ Plot the grid frequencx history.

        :param: gridHistory - instanceof the grid history model
        """
        assert(type(gridHistory == EnfModel))

        data = gridHistory.getData()
        pen = pg.mkPen(color=(0, 255, 0))

        if self.gridFreqCurve:
            self.plotWidget.removeItem(self.gridFreqCurve)
        self.gridFreqCurve = self.plotWidget.plot(name="Grid frequence history", pen=pen)
        self.gridFreqCurve.setData(list(range(len(data))), data)


    def showMatches(self, matches):
        # print("Show matches")
        self.e_result.setText(str(matches[0][0]))


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

        self.unsetCursor()


    def onLoadGridHistory(self):
        self.setCursor(Qt.WaitCursor)

        location = self.l_country.currentText()
        year = int(self.l_year.currentText())
        month = self.l_month.currentIndex()
        self.controller.onLoadGridHistory(location, year, month)
        #self.gridHistory = GridHistoryModel(location, year, month)

        self.unsetCursor()


    def onAnalyse(self):
        self.setCursor(Qt.WaitCursor)

        harmonic = int(self.b_harmonic.value())
        self.controller.onAnalyse()

        self.unsetCursor()


class HumController(QApplication):
    """ Orchestrate view and model. """
    def __init__(self, argv):
        super(HumController, self).__init__(argv)
        self.view = HumView(self)
        self.model = None

    def show(self):
        self.view.show()

    def loadAudioFile(self, fileName):
        """ Create a model from an audio recoding and tell the view to show it."""
        self.model = EnfModel()
        self.model.fromWaveFile(fileName)
        self.model.makeEnf(2)
        self.view.plotAudioRec(self.model)

    def onLoadGridHistory(self, location, year, month):
        self.gm = EnfModel()
        self.gm.fromWaveFile("71000_ref.wav")
        self.gm.makeEnf()
        self.view.plotGridHistory(self.gm)

    def onAnalyse(self):
        if self.model and self.gm:
            m = self.model.match(self.gm)
            self.view.showMatches(m)
            self.view.plotAudioRec(self.model, t_offset=m[0][0])


#
# Main
#
if __name__ == '__main__':
    ######### Test
    #enf = EnfModel()
    #enf.fromWaveFile("001.wav")
    #enf.makeEnf(2)
    #ref = EnfModel()
    #ref.fromWaveFile("71000_ref.wav")
    #ref.makeEnf()
    #enf.match(ref)
    ######### End test
    app = HumController([])
    app.show()
    app.exec()
