#
# Skeleton for an application for video analysis. Uses Qt video player.
#

import sys
from PyQt5.QtCore import QDir, Qt, QUrl
import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
                             QVBoxLayout, QLineEdit, QFileDialog, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QCheckBox,
                             QComboBox, QSpinBox, QTabWidget, QDoubleSpinBox,
                             QMenuBar, QAction, QDialog, QMessageBox,
                             QDialogButtonBox, QProgressDialog, QHBoxLayout,
                             QStyle, QSlider, QSizePolicy)
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.Qt import Qt
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIcon

from griddata import GridDataAccessFactory


class FlimmerView(QMainWindow):

    # Colour definitions
    # Order is R, G, B, alpha
    regionAreaPen = pg.mkPen(color=(10, 10, 80))
    regionAreaBrush = pg.mkBrush(color=(240, 240, 240, 128))
    regionAreaHoverBrush = pg.mkBrush(color=(200, 200, 200, 128))
    spectrumCurveColour = pg.mkPen(color=(255, 0, 0))
    ENFvalueColour = pg.mkPen(color=(150, 0, 0))
    ENFsmoothedValueColour = pg.mkPen(color=(153, 153, 0))
    GridCurveColour = pg.mkPen(color=(0, 150, 70))
    correlationCurveColour = pg.mkPen(color=(255, 0, 255))

    month_names = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


    def __init__(self, parent=None):
        super().__init__(parent)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        #self.videoWidgets()
        self.__createWidgets()


    def videoWidgets(self):
        """Create a layout with the video-related widgets.

        :returns layout: a QLayout with the widgets added.
        """
        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                QSizePolicy.Maximum)

        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open movie')
        openAction.triggered.connect(self.__onOpenFileClicked)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        #fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(exitAction)

        # Create a widget for window contents
        #wid = QWidget(self)
        #self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget, stretch=1)
        layout.addLayout(controlLayout)
        layout.addWidget(self.errorLabel)

        # Set widget to contain window contents
        #wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)

        return layout


    def __createWidgets(self):
        """Create widgets including curves and legends for the plot widgets."""
        widget = QWidget()
        self.setWindowTitle("Flimmer")

        # Define layouts
        main_layout = QHBoxLayout()
        left_layout = self.videoWidgets()
        main_layout.addLayout(left_layout)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)
        audio_area = QGridLayout()
        audio_group = QGroupBox("Video")
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
                                           pen=FlimmerView.spectrumCurveColour)
        self.tabs.addTab(self.clipSpectrumPlot, "Clip Spectrum")

        # Create a plot widget for the various ENF curves and add it to the
        # tab
        self.enfPlot = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.enfPlot.setLabel("left", "Frequency (mHz)")
        self.enfPlot.setLabel("bottom", "Date and time")
        self.enfPlot.addLegend()
        self.enfPlot.setBackground("w")
        self.enfPlot.showGrid(x=True, y=True)
        self.enfPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        self.enfAudioCurve = self.enfPlot.plot(name="Clip ENF values",
                                               pen=FlimmerView.ENFvalueColour)
        self.enfAudioCurveSmothed = self.enfPlot.plot(name="Smoothed clio ENF values",
                                               pen=FlimmerView.ENFsmoothedValueColour)
        self.enfGridCurve = self.enfPlot.plot(name="Grid frequency history",
                                               pen=FlimmerView.GridCurveColour)
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
                                                   pen=FlimmerView.correlationCurveColour)
        self.tabs.addTab(self.correlationPlot, "Correlation")

        # 'audio' area
        self.b_load = QPushButton("Load")
        self.b_load.setToolTip("Load a WAV file to analyse.")
        self.b_load.clicked.connect(self.__onOpenFileClicked)
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

        analyse_area.setColumnStretch(7, 1)

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
        self.l_year0 = QComboBox(self)
        for y in range(2024, 2000 - 1, -1):
            self.l_year0.addItem(f'{y}')
        grid_area.addWidget(self.l_year0, 0, 3)
        grid_area.addWidget(QLabel("Month"), 0, 4)
        self.l_month0 = QComboBox()
        self.l_month0.addItems(FlimmerView.month_names)
        grid_area.addWidget(self.l_month0, 0, 5)

        grid_area.addWidget(QLabel("Year"), 1, 2)
        self.l_year1 = QComboBox(self)
        for y in range(2024, 2000 - 1, -1):
            self.l_year1.addItem(f'{y}')
        grid_area.addWidget(self.l_year1, 1, 3)
        grid_area.addWidget(QLabel("Month"), 1, 4)
        self.l_month1 = QComboBox()
        self.l_month1.addItems(FlimmerView.month_names)
        grid_area.addWidget(self.l_month1, 1, 5)

        self.b_loadGridHistory = QPushButton("Load")
        grid_area.addWidget(self.b_loadGridHistory, 2, 0)
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

        # Overall layout
        right_layout.addWidget(self.tabs, 1)
        right_layout.addWidget(audio_group)
        right_layout.addWidget(analyse_group)
        right_layout.addWidget(grid_group)
        right_layout.addWidget(result_group)

        #self.__setButtonStatus()

        widget.setLayout(main_layout)
        self.setCentralWidget(widget)


    #def listPlayerInfo(self):
    #    self.mediaPlayer.hasSupport(mimeType, codecs, flags)


    def __onOpenFileClicked(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie",
                QDir.homePath())

        if fileName != '':
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
            self.playButton.setEnabled(True)


    def __onAnalyseClicked(self):
        pass


    def __onLoadGridHistoryClicked(self):
        pass


    def __onMatchClicked(self):
        pass


    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))


    def positionChanged(self, position):
        self.positionSlider.setValue(position)


    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)


    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)


    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())


    def exitCall(self):
        pass


class FlimmerApp(QApplication):
    """ Create a FlimmerView object and show it. """
    def __init__(self, argv):
        super(FlimmerApp, self).__init__(argv)
        self.view = FlimmerView()

    def show(self):
        self.view.show()


if __name__ == '__main__':
    try:
        #app = FlimmerApp([])
        app = QApplication(sys.argv)
        player = FlimmerView()
        #player.resize(1200, 1000)
        player.show()
        #app.exec()
        sys.exit(app.exec_())
    except MemoryError as e:
        print(e)
