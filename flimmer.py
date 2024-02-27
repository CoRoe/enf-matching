#
# Skeleton for an application for video analysis. Uses Qt video player.
#

import sys
import json
import os
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
from enf import VideoEnf


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
        self.clip = None
        self.grid = None
        self.settings = Settings()
        self.databasePath = self.settings.databasePath()

        self.enfAudioCurve = None     # ENF series of loaded audio file
        self.enfAudioCurveSmothed = None
        self.enfAudioCurveRegion = None
        self.clipSpectrumCurve = None # Fourier transform of loaded audio file
        self.enfGridCurve = None      # ENF series of grid
        self.correlationCurve = None  # Correlation of ENF series of audio
                                      # clip and grid

        self.__createWidgets()


    def videoWidgets_unused(self):
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
        #left_layout = self.videoWidgets()
        #main_layout.addLayout(left_layout)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)
        video_area = QGridLayout()
        audio_group = QGroupBox("Video")
        audio_group.setLayout(video_area)
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

        # 'Video' area
        self.b_load = QPushButton("Load")
        self.b_load.setToolTip("Load a video file to analyse.")
        self.b_load.clicked.connect(self.__onOpenFileClicked)
        video_area.addWidget(self.b_load, 0, 0)
        self.e_fileName = QLineEdit()
        self.e_fileName.setReadOnly(True)
        self.e_fileName.setToolTip("Video file that has been loaded.")
        video_area.addWidget(self.e_fileName, 0, 1, 1, 3)
        video_area.addWidget(QLabel("Video format"), 1, 0)
        self.e_videoFormat = QLineEdit()
        self.e_videoFormat.setReadOnly(True)
        video_area.addWidget(self.e_videoFormat, 1, 1)
        video_area.addWidget(QLabel("Frame rate"), 1, 2)
        self.e_frameRate = QLineEdit()
        self.e_frameRate.setReadOnly(True)
        video_area.addWidget(self.e_frameRate, 1, 3)
        video_area.addWidget(QLabel("Duration (sec)"), 1, 4)
        self.e_duration = QLineEdit()
        self.e_duration.setReadOnly(True)
        video_area.addWidget(self.e_duration, 1, 5)
        video_area.setColumnStretch(5, 1)

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


    def __setRegion(self, region, movable=True):
        """Set the region of interest.

        :param region: A tuple specifying start and end of the region of interest.
        """
        if self.enfAudioCurveRegion is not None:
            self.enfPlot.removeItem(self.enfAudioCurveRegion)
        self.enfAudioCurveRegion = pg.LinearRegionItem(values=region,
                                                       pen=FlimmerView.regionAreaPen,
                                                       bounds=region)
        self.enfAudioCurveRegion.setBrush(FlimmerView.regionAreaBrush)
        self.enfAudioCurveRegion.setHoverBrush(FlimmerView.regionAreaHoverBrush)
        self.enfAudioCurveRegion.sigRegionChangeFinished.connect(self.__onRegionChanged)
        self.enfAudioCurveRegion.setMovable(movable)
        self.enfPlot.addItem(self.enfAudioCurveRegion)



    #def listPlayerInfo(self):
    #    self.mediaPlayer.hasSupport(mimeType, codecs, flags)


    def __onOpenFileClicked(self):
        self.setCursor(Qt.WaitCursor)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, x = QFileDialog.getOpenFileName(self,"Open video file",
                                                  "*.mp4", "all files (*)",
                                                  options=options)
        if fileName and fileName != '':
            self.clip = VideoEnf(self.enfAudioCurve, self.enfAudioCurveSmothed,
                                self.clipSpectrumCurve)
            videoProp = self.clip.getVideoProperties(fileName)
            if videoProp is not None:
                self.e_fileName.setText(fileName)
                self.e_duration.setText(str(self.clip.getDuration()))
                self.e_frameRate.setText(str(self.clip.getFrameRate()))
                self.e_videoFormat.setText(self.clip.getVideoFormat())
                self.clip.loadVideoFile(fileName)

                # Clear all clip-related plots and the region
                if self.enfAudioCurveRegion:
                    self.enfPlot.removeItem(self.enfAudioCurveRegion)
                    self.enfAudioCurveRegion = None
            else:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Data Error")
                dlg.setIcon(QMessageBox.Information)
                dlg.setText(f"Could not handle {fileName}. Maybe it is not a"
                            " video file.")
                dlg.exec()

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
        else:
            self.clip.clearSmoothedENF()
        self.clip.makeFFT()
        if self.grid is not None:
            gridtimestamp = self.grid.getTimestamp()
            self.clip.setTimestamp(gridtimestamp)

        # Set range of the x axis to the clip length
        t = self.clip.getTimestamp()
        self.enfPlot.setXRange(t, t + self.clip.clip_len_s)

        # Plot curves
        self.clip.plotENF()
        self.clip.plotENFsmoothed()
        self.clip.plotSpectrum()

        # Display region; initially, it comprises the whole clip
        rgn = self.clip.getENFRegion()
        self.__setRegion(rgn)

        self.unsetCursor()
        self.tabs.setCurrentIndex(1)
        self.__setButtonStatus()


    def __onLoadGridHistoryClicked(self):
        pass


    def __onMatchClicked(self):
        pass


    @pyqtSlot()
    def __onRegionChanged(self):
        """Called when the user has dragged one of the region boundaries.

        Queries the actual region boundaries from the plot widget and
        sets the region in the clip.
        """
        rgn = self.enfAudioCurveRegion.getRegion()
        self.clip.setENFRegion(rgn)


    def __setButtonStatus(self):
        """ Enables or disables buttons depending on the clip status."""
        audioDataLoaded = self.clip is not None and self.clip.fileLoaded()
        audioEnfLoaded = self.clip is not None and self.clip.ENFavailable()
        gridEnfLoaded = self.grid is not None and self.grid.ENFavailable()

        self.b_analyse.setEnabled(audioDataLoaded)
        self.b_match.setEnabled(audioEnfLoaded and gridEnfLoaded)



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
