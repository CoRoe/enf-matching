#
# flimmer.py: Extract ENF components from video recordings and match them
# against databases of historical grid frequency deviations.
#
# Copyright (C) 2024 conrad.roeber@mailbox.org
#

import sys
import pyqtgraph as pg
from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
                             QVBoxLayout, QLineEdit, QFileDialog, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QCheckBox,
                             QComboBox, QSpinBox, QTabWidget, QDoubleSpinBox,
                             QMenuBar, QAction, QDialog, QMessageBox,
                             QDialogButtonBox, QProgressDialog)
from PyQt5.Qt import Qt, QSettings, QFileInfo, QRadioButton
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5 import QtGui
import numpy as np

import datetime
from griddata import GridDataAccessFactory
from enf import VideoClipEnf, GridEnf


class FlimmerView(QMainWindow):
    """Handle display and user interaction.
    """

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

    class GetGridDataWorker(QObject):
        """Worker task to load ENF data from either the internet or -- if already cached --
        from a databse. Is intended to run in QThread and communicates via signals."""

        finished = pyqtSignal()
        progress = pyqtSignal(str, int)

        def __init__(self, grid, location, year, month, n_months,
                     progressCallback):
            """Initialise the worker object.

            :param grid: GridEnf object for wich to fetch the frequency data.
            :param location: Grid location
            :param year: The year for whitch the data should be fetched.
            :param month: The first month.
            :param n_months: The number of consecutive months.
            :param progressCallback: A function to report progress to the
            caller.

            Just stores all parameters in instance variables for use in the
            'run' method.
            """
            super().__init__()
            self.grid = grid
            self.location = location
            self.year = year
            self.month = month
            self.n_months = n_months
            self.progressCallback = progressCallback

        @pyqtSlot()
        def run(self):
            """Delegate the task to the 'grid' object. When finished send
            a 'finished' signal."""
            print("GetGridDataWorker.run()")
            self.grid.loadGridEnf(self.location, self.year, self.month, self.n_months,
                                  self.__on_progress)
            self.finished.emit()

        def __on_progress(self, hint, progr):
            """Send a 'progress' signal. The method is called from the underlying
            loadGridEnf() method."""
            self.progress.emit(hint, progr)


    month_names = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


    def __init__(self, parent=None):
        super().__init__(parent)
        self.clip = None
        self.grid = None
        # self.settings = Settings()
        self.__qsettings = QSettings("CRO", "Flimmer")
        # self.databasePath = self.settings.databasePath()

        self.__enfAudioCurve = None     # ENF series of loaded audio file
        self.__enfSmoothedAudioCurve = None
        self.enfAudioCurveRegion = None
        self.__clipSpectrumCurve = None # Fourier transform of loaded audio file
        self.__enfGridCurve = None      # ENF series of grid
        self.__correlationCurve = None  # Correlation of ENF series of audio
                                      # clip and grid

        self.__createWidgets()
        self.__createMenu()
        self.__loadSettings()

        # Trick to enable / disable the widgets related to the video
        # processing mode.
        self.__adjustVideoModeWidgets()


    def __loadSettings(self):
        finfo = QFileInfo(self.__qsettings.fileName())

        if finfo.exists() and finfo.isFile():
            for w in self.findChildren(QSpinBox):
                if w.objectName() != "":
                    value = self.__qsettings.value(w.objectName())
                    if value is not None:
                        w.setValue(int(value))
            for w in self.findChildren(QComboBox):
                if w.objectName() != "":
                    value = self.__qsettings.value(w.objectName())
                    if value is not None:
                        w.setCurrentText(value)
            for w in self.findChildren(QRadioButton):
                if w.objectName() != "":
                    value = self.__qsettings.value(w.objectName())
                    if value is not None:
                        print(w.objectName(), '->', value)
                        w.setChecked(value == 'true')

        # Set default
        self.databasePath = self.__qsettings.value("paths/database")
        if self.databasePath is None:
            self.databasePath = "/tmp/hum.sqlite"
            self.__qsettings.setValue("paths/database",self.databasePath)


    def __saveSettings(self):
        #print(self.__qsettings.fileName())
        for w in self.findChildren(QSpinBox):
            if w.objectName() != "":
                self.__qsettings.setValue(w.objectName(), w.value())
        for w in self.findChildren(QComboBox):
            if w.objectName() != "":
                self.__qsettings.setValue(w.objectName(), w.currentText())
        for w in self.findChildren(QRadioButton):
            if w.objectName() != "":
                self.__qsettings.setValue(w.objectName(), w.isChecked())



    def __createVideoGroupWidgets(self):
        """Create all widgets releated to video file loading and display
        of video properties.

        :returns video_group: The QGroupBox containing the other widgets.
        """
        video_area = QGridLayout()
        video_group = QGroupBox("Video")
        video_group.setLayout(video_area)

        # 'Video' area
        self.b_rolling_shutter = QRadioButton("Rolling Shutter", objectName="Rolling-shutter")
        video_area.addWidget(self.b_rolling_shutter, 0, 0)
        self.b_rolling_shutter.clicked.connect(self.__onAnalyseRSClicked)
        video_area.addWidget(QLabel("Sensor read-out time"), 0, 1)
        self.sp_readOutTime = QSpinBox(objectName='readout-time')
        self.sp_readOutTime.setRange(0, 50)
        video_area.addWidget(self.sp_readOutTime, 0, 2)

        self.b_grid_roi = QRadioButton("Grid ROI", objectName="GridROI")
        video_area.addWidget(self.b_grid_roi, 1, 0)
        self.b_grid_roi.clicked.connect(self.__onAnalyseGridROIClicked)

        video_area.addWidget(QLabel("# Grid rows:"), 1, 1)
        self.sp_vert = QSpinBox(objectName="gridroi-vertical")
        self.sp_vert.setRange(1, 10)
        video_area.addWidget(self.sp_vert, 1, 2)
        video_area.addWidget(QLabel("# Grid columns:"), 1, 3)
        self.sp_horiz = QSpinBox(objectName="gridroi-horizontal")
        self.sp_horiz.setRange(1, 10)
        video_area.addWidget(self.sp_horiz, 1, 4)

        self.b_load = QPushButton("Load")
        self.b_load.setToolTip("Load a video file to analyse.")
        self.b_load.clicked.connect(self.__onOpenFileClicked)
        video_area.addWidget(self.b_load, 2, 0)
        self.e_fileName = QLineEdit()
        self.e_fileName.setReadOnly(True)
        self.e_fileName.setToolTip("Video file that has been loaded.")
        video_area.addWidget(self.e_fileName, 2, 1, 1, 5)

        video_area.addWidget(QLabel("Video format"), 3, 0)
        self.e_videoFormat = QLineEdit()
        self.e_videoFormat.setReadOnly(True)
        video_area.addWidget(self.e_videoFormat, 3, 1)
        video_area.addWidget(QLabel("Frame rate"), 3, 2)
        self.e_frameRate = QLineEdit()
        self.e_frameRate.setReadOnly(True)
        video_area.addWidget(self.e_frameRate, 3, 3)
        video_area.addWidget(QLabel("Duration (sec)"), 3, 4)
        self.e_duration = QLineEdit()
        self.e_duration.setReadOnly(True)
        video_area.addWidget(self.e_duration, 3, 5)


        video_area.setColumnStretch(6, 1)

        return video_group


    def __createAnalyseGroupWidgets(self):
        """Create all widgets related to video file analysis.

        :returns analyse_group: The QGroupBox containing the other widgets.
        """
        analyse_group = QGroupBox("Analysis")
        analyse_area = QGridLayout()
        analyse_group.setLayout(analyse_area)

        # Add subgroups
        #analyse_area.addWidget(self.__createAnalyseCommon())
        #analyse_area.addWidget(self.__createAnalyseRSGroup())
        #analyse_area.addWidget(self.__createAnalyseGridROIGroup())
        #analyse_area.addWidget(self.__createAnalyseOutliers())

        analyse_area.addWidget(QLabel("Grid freq"), 0, 0)
        self.b_nominal_freq = QComboBox(objectName='grid-freq')
        self.b_nominal_freq.addItems(("50", "60"))
        self.b_nominal_freq.setToolTip("The nominal frequency of the power grid at the place of the recording;"
                                       " 50 Hz in most countries.")
        analyse_area.addWidget(self.b_nominal_freq, 0, 1)
        analyse_area.addWidget(QLabel("Harmonic"), 0, 2)
        self.cb_gridHarmonic = QComboBox(objectName="Grid-Harmonic")
        self.cb_gridHarmonic.addItems(("1", "2", "3"))
        self.cb_gridHarmonic.currentIndexChanged.connect(self.__setButtonStatus)
        analyse_area.addWidget(self.cb_gridHarmonic, 0, 3)
        analyse_area.addWidget(QLabel("Vid. harmonic"), 0, 4)
        self.cb_enfHarmonic = QComboBox(objectName="Vid-Frame-Harmonics")
        self.cb_enfHarmonic.addItems(("-3", "-2", "-1", "1", "2", "3"))
        self.cb_enfHarmonic.currentIndexChanged.connect(self.__setButtonStatus)
        analyse_area.addWidget(self.cb_enfHarmonic, 0, 5)
        analyse_area.addWidget(QLabel("Alias freq."), 0, 6)
        self.le_aliasFreq = QLineEdit()
        analyse_area.addWidget(self.le_aliasFreq, 0, 7)
        self.le_aliasFreq.setEnabled(False)
        analyse_area.addWidget(QLabel("Band width"), 0, 8)
        self.b_band_size = QSpinBox(objectName='bandwidth')
        self.b_band_size.setRange(0, 1000)
        self.b_band_size.setValue(200)
        self.b_band_size.setMinimumWidth(100)
        self.b_band_size.setSuffix(" mHz")
        analyse_area.addWidget(self.b_band_size, 0, 9)

        self.c_rem_outliers = QCheckBox("Rm. outliers")
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

        # 'Analyse' button in the last row
        self.b_analyse = QPushButton("Analyse")
        self.b_analyse.clicked.connect(self.__onAnalyseClicked)
        analyse_area.addWidget(self.b_analyse, 2, 0)

        analyse_area.setColumnStretch(10, 1)

        return analyse_group


    def __createGridAreaWidgets(self):
        """Create all widgets related to loading grid frequencies.

        :returns grid_group: QGroupBox containing the other widgets.
        """
        grid_group = QGroupBox("Grid")
        grid_area = QGridLayout()
        grid_group.setLayout(grid_area)
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

        return grid_group


    def __createResultAreaWidgets(self):
        """Create the widgets releated to displaying the analysis result.

        :returns result_group: QGroupBox containing the other widgets.
        """
        result_group = QGroupBox("Result")
        result_area = QGridLayout()
        result_group.setLayout(result_area)

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

        result_area.setColumnStretch(6, 1)

        return result_group


    def __createWidgets(self):
        """Create widgets including curves and legends for the plot widgets."""
        widget = QWidget()
        self.setWindowTitle("Flimmer")

        # Define layouts
        main_layout = QVBoxLayout()

        self.tabs = QTabWidget()

        #
        # Spectrogram
        #

        # Create a layout window that will contain an image and histogram
        self.spectrogr_container = pg.GraphicsLayoutWidget()
        self.spectrogr_container.setBackground('w')

        # Item for displaying image data
        # A plot area (ViewBox + axes) for displaying the image
        self.spectrogr_plot = self.spectrogr_container.addPlot()

        # Item for displaying image data
        # Interpret image data as row-major instead of col-major
        self.spectorgr_img = pg.ImageItem(axisOrder='row-major')
        self.spectrogr_plot.addItem(self.spectorgr_img)

        # Add a histogram with which to control the gradient of the image
        self.spectogr_hist = pg.HistogramLUTItem()

        # This gradient is roughly comparable to the gradient used by Matplotlib
        # You can adjust it and then save it using spectogr_hist.gradient.saveState()
        self.spectogr_hist.gradient.restoreState(
                {'mode': 'rgb',
                 'ticks': [(0.0, (0, 128, 128, 255)),
                           (0.5, (255, 0, 0, 255)),
                           (1.0, (255, 255, 0, 255))
                           ]})

        # Link the histogram to the image
        self.spectogr_hist.setImageItem(self.spectorgr_img)

        # If you don't add the histogram to the window, it stays invisible, but I find it useful.
        self.spectrogr_container.addItem(self.spectogr_hist)

        # Add labels to the axis
        self.spectrogr_plot.setLabel('bottom', "Time", units='s')

        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in this case kHz)
        self.spectrogr_plot.setLabel('left', "Frequency", units='Hz')
        self.tabs.addTab(self.spectrogr_container, "Clip Spectrogram")

        #
        # Create a plot widget for the video clip spectrum and add it to the
        # tab
        #
        self.clipSpectrumPlot = pg.PlotWidget()
        self.clipSpectrumPlot.setLabel("left", "Amplitude")
        self.clipSpectrumPlot.setLabel("bottom", "Frequency", unit="Hz")
        self.clipSpectrumPlot.addLegend()
        self.clipSpectrumPlot.setBackground("w")
        self.clipSpectrumPlot.showGrid(x=True, y=True)
        self.clipSpectrumPlot.setXRange(0, 1000)
        self.clipSpectrumPlot.plotItem.setMouseEnabled(
            y=False
        )  # Only allow zoom in X-axis
        self.__clipSpectrumCurve = self.clipSpectrumPlot.plot(
            name="WAV file spectrum", pen=FlimmerView.spectrumCurveColour
        )
        self.tabs.addTab(self.clipSpectrumPlot, "Clip Spectrum")

        # Create a plot widget for the audio clip spectrum and add it to the
        # tab
        #self.clipSpectrumPlot = pg.PlotWidget()
        #self.clipSpectrumPlot.setLabel("left", "Amplitude")
        #self.clipSpectrumPlot.setLabel("bottom", "Frequency (Hz)")
        #self.clipSpectrumPlot.addLegend()
        #self.clipSpectrumPlot.setBackground("w")
        #self.clipSpectrumPlot.showGrid(x=True, y=True)
        #self.clipSpectrumPlot.setXRange(0, 1000)
        #self.clipSpectrumPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        #self.__clipSpectrumCurve = self.clipSpectrumPlot.plot(name="Clip spectrum",
        #                                   pen=FlimmerView.spectrumCurveColour)
        #self.tabs.addTab(self.clipSpectrumPlot, "Clip Spectrum")

        # Create a plot widget for the various ENF curves and add it to the
        # tab
        self.enfPlot = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.enfPlot.setLabel("left", "Frequency (mHz)")
        self.enfPlot.setLabel("bottom", "Date and time")
        self.enfPlot.addLegend()
        self.enfPlot.setBackground("w")
        self.enfPlot.showGrid(x=True, y=True)
        self.enfPlot.plotItem.setMouseEnabled(y=False) # Only allow zoom in X-axis
        self.__enfAudioCurve = self.enfPlot.plot(name="Clip ENF values",
                                               pen=FlimmerView.ENFvalueColour)
        self.__enfSmoothedAudioCurve = self.enfPlot.plot(name="Smoothed clio ENF values",
                                               pen=FlimmerView.ENFsmoothedValueColour)
        self.__enfGridCurve = self.enfPlot.plot(name="Grid frequency history",
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
        self.__correlationCurve = self.correlationPlot.plot(name="Correlation",
                                                   pen=FlimmerView.correlationCurveColour)
        self.tabs.addTab(self.correlationPlot, "Correlation")

        # Overall layout
        main_layout.addWidget(self.tabs, 1)
        main_layout.addWidget(self.__createVideoGroupWidgets())
        main_layout.addWidget(self.__createAnalyseGroupWidgets())
        main_layout.addWidget(self.__createGridAreaWidgets())
        main_layout.addWidget(self.__createResultAreaWidgets())

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


    def __editSettings(self):
        """Menu item; pops up a 'setting' dialog."""
        dlg = SettingsDialog(self.__qsettings)
        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")


    def __checkDateRange(self):
        """Check if 'to' date is later than 'from' date and computes the
        number of months between 'from' and 'to' date.

        :returns year0: The 'from' year
        :returns month0: The 'from'month (1..12)
        :returns n_months: The number of months
        """
        year0 = int(self.l_year0.currentText())
        month0 = self.l_month0.currentIndex()
        year1 = int(self.l_year1.currentText())
        month1 = self.l_month1.currentIndex()
        n_months = (year1 * 12 + month1) - (year0 * 12 + month0) + 1
        print(f"Get grid frequencies from {year0}-{month0+1:02} to {year1}-{month1+1:02}, {n_months} months")
        return year0, month0 + 1, n_months


    def __showEnfSources(self):
        self.setCursor(Qt.WaitCursor)
        dlg = ShowEnfSourcesDlg(self)
        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")
        self.unsetCursor()


    @pyqtSlot()
    def __setButtonStatus(self):
        """Enable or disable buttons depending on the clip status and alias frequency settings."""

        if self.clip is not None:
            gh = int(self.cb_gridHarmonic.currentText())
            vh = int(self.cb_enfHarmonic.currentText())
            grid_freq = int(self.b_nominal_freq.currentText())
            aliasFreq = self.clip.aliasFreqs(grid_freq, gh, vh)
            if aliasFreq is None:
                self.le_aliasFreq.setText('Invalid')
            else:
                self.le_aliasFreq.setText(str(aliasFreq))
        else:
            aliasFreq = None
            self.le_aliasFreq.setText("")

        audioDataLoaded = self.clip is not None and self.clip.fileLoaded()
        audioEnfLoaded = self.clip is not None and self.clip.ENFavailable()
        gridEnfLoaded = self.grid is not None and self.grid.ENFavailable()

        self.b_analyse.setEnabled(audioDataLoaded and aliasFreq is not None)
        self.b_match.setEnabled(audioEnfLoaded and gridEnfLoaded)


    @pyqtSlot()
    def __onAnalyseRSClicked(self):
        """Radio button clicked."""
        print("__onAnalyseRSClicked()")
        s = self.b_rolling_shutter.isChecked()
        self.b_grid_roi.setChecked(not s)
        self.__adjustVideoModeWidgets()


    @pyqtSlot()
    def __onAnalyseGridROIClicked(self):
        """Radio button clicked."""
        print("__onAnalyseGridROIClicked()")
        s = self.b_grid_roi.isChecked()
        self.b_rolling_shutter.setChecked(not s)
        self.__adjustVideoModeWidgets()


    def __onOpenFileClicked(self):
        """Button to open a multimedia file clicked."""
        self.setCursor(Qt.WaitCursor)

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, x = QFileDialog.getOpenFileName(self,"Open video file",
                                                  "*.mp4", "all files (*)",
                                                  options=options)
        if fileName and fileName != '':
            self.clip = VideoClipEnf()
            videoProp = self.clip.getVideoProperties(fileName)
            if videoProp is not None:
                self.e_fileName.setText(fileName)
                self.e_duration.setText(str(self.clip.getDuration()))
                self.e_frameRate.setText(str(self.clip.getFrameRate()))
                self.e_videoFormat.setText(self.clip.getVideoFormat())

                # File loading depends on the analyse method
                if self.b_grid_roi.isChecked():
                    self.clip.loadVideoFileGridROI(fileName,
                                                  self.sp_vert.value(), self.sp_horiz.value())
                else:
                    self.clip.loadVideoFileRollingShutter(fileName, self.sp_readOutTime.value())
                self.__setButtonStatus()

                # Clear all clip-related plots and the region
                if self.enfAudioCurveRegion:
                    self.enfPlot.removeItem(self.enfAudioCurveRegion)
                    self.enfAudioCurveRegion = None

                # Plot video spectrum
                self.__plotClipSpectrogram()
                self.__plotClipSpectrum()
            else:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Data Error")
                dlg.setIcon(QMessageBox.Warning)
                dlg.setText(f"Could not handle {fileName}. Maybe it is not a"
                            " video file.")
                dlg.exec()

        self.unsetCursor()
        self.__setButtonStatus()


    @pyqtSlot()
    def __onAnalyseClicked(self):
        """ Called when the 'analyse' button is pressed. """
        # Display wait cursor
        self.setCursor(Qt.WaitCursor)

        self.clip.makeEnf(int(self.b_nominal_freq.currentText()),
            int(self.le_aliasFreq.text()),
            int(self.cb_gridHarmonic.currentText()),
            float(self.b_band_size.value()/1000),
            int(self.cb_enfHarmonic.currentText()),
            30)

        if self.c_rem_outliers.isChecked():
            m = self.sp_Outlier_Threshold.value()
            window = self.sp_window.value()
            self.clip.outlierSmoother(m, window)
        else:
            self.clip.clearSmoothedENF()
        # self.clip.makeFFT()

        if self.grid is not None:
            gridtimestamp = self.grid.getTimestamp()
            self.clip.setTimestamp(gridtimestamp)

        # Set range of the x axis to the clip length
        t = self.clip.getTimestamp()
        self.enfPlot.setXRange(t, t + self.clip.clip_len_s)

        # Plot curves. In some cases it is not possible to extract the
        # the ENF values, hence the check.
        if self.clip.ENFavailable():
            #self.clip.plotENF()
            #self.clip.plotENFsmoothed()
            self.__plotClipEnf()
        #self.__plotClipSpectrogram()
        #self.__plotClipSpectrum()

        # Display region; initially, it comprises the whole clip
        rgn = self.clip.getENFRegion()
        self.__setRegion(rgn)

        self.unsetCursor()
        self.tabs.setCurrentIndex(1)
        self.__setButtonStatus()


    def __adjustVideoModeWidgets(self):

        s = self.b_rolling_shutter.isChecked()
        self.sp_readOutTime.setEnabled(s)
        self.sp_horiz.setEnabled(not s)
        self.sp_vert.setEnabled(not s)

        s = self.b_grid_roi.isChecked()
        self.sp_readOutTime.setEnabled(not s)
        self.sp_horiz.setEnabled(s)
        self.sp_vert.setEnabled(s)


    @pyqtSlot()
    def __onLoadGridHistoryClicked(self):
        """Gets historical ENF values from an ENF database. Called when the
        'load' button in the 'grid' field is clicked.

        The method checks year and month setting: The 'to' data must be later
        than the 'from' date; they must not span more than 12 months. This
        condition exists to limit processing time and memory.

        The method then starts a worker thread to get the ENF data -- either
        from a local database or from the internet. During this operaton, a
        progress dialog is displayed. Any data downloaded from the internet is
        then saved in the database.

        Signals are used to coordinate the GUI thread and the worker thread
        and the progress dialog.

        :See __onLoadGridHistoryDone():

        """
        # Some description of QT threads:
        # https://gist.github.com/majabojarska/952978eb83bcc19653be138525c4b9da

        location = self.l_country.currentText()
        year, month, n_months = self.__checkDateRange()
        self.grid = GridEnf(self.databasePath)

        if location == 'Test':
            self.setCursor(Qt.WaitCursor)
            self.grid.loadWaveFile("71000_ref.wav")
            self.grid.makeEnf(int(self.b_nominal_freq.currentText()),
                            float(self.b_band_size.value()/1000),
                            int(self.cb_gridHarmonic.currentText()))
            #self.grid.plotENF()
            self.__plotGridEnf()
            self.tabs.setCurrentIndex(2)
            self.unsetCursor()
            self.__setButtonStatus()
        else:
            if n_months < 1:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Error")
                dlg.setIcon(QMessageBox.Information)
                dlg.setText(f"'To' date must be later than 'from' date")
                dlg.exec()
            elif n_months > 12:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Error")
                dlg.setIcon(QMessageBox.Information)
                dlg.setText(f"Limit are 12 months")
                dlg.exec()
            else:
                self.__ldGridProgDlg = QProgressDialog("Loading ENF data from inrternet", "Cancel",
                                                     0, n_months, self)
                self.__ldGridProgDlg.setWindowTitle("Getting ENF data")
                self.__ldGridProgDlg.setCancelButtonText(None)
                self.__ldGridProgDlg.setWindowModality(Qt.WindowModal)
                self.__ldGridProgDlg.forceShow()
                self.__ldGridProgDlg.setValue(0)

                # Move to thread
                self.__loadGridEnfThread = QThread()
                self.__loadGridEnfWorker = FlimmerView.GetGridDataWorker(self.grid,
                                                           location, year, month, n_months,
                                                           self.__gridHistoryLoadingProgress)
                self.__loadGridEnfWorker.moveToThread(self.__loadGridEnfThread)

                # Connect signale
                self.__loadGridEnfThread.started.connect(self.__loadGridEnfWorker.run)
                self.__loadGridEnfThread.finished.connect(self.__loadGridEnfThread.deleteLater)
                self.__loadGridEnfWorker.finished.connect(self.__loadGridEnfThread.quit)
                self.__loadGridEnfWorker.finished.connect(self.__onLoadGridHistoryDone)
                self.__loadGridEnfWorker.progress.connect(self.__gridHistoryLoadingProgress)
                self.__loadGridEnfThread.start()



    @pyqtSlot()
    def __onLoadGridHistoryDone(self):
        """Called when __loadGridEnfWorker() finishes.

        It sets the timestamp of the clip (which was previously 0)
        to the timestamp of the grid data. This way, the clip curve appears
        to be at the grid curve.

        If there no clip yet then the displayed timespan is the timespan
        of the grid data; otherwise set the timespan to the one of the
        clip.
        """
        print("__onLoadGridHistoryDone")

        # Terminate thread and wait
        self.__loadGridEnfThread.quit()
        self.__loadGridEnfThread.wait()

        self.__ldGridProgDlg.cancel()
        if self.grid.enf is not None:
            self.setCursor(Qt.WaitCursor)
            if self.clip is not None:
                # Set the clip's timestamp to the grid data timestamp.
                self.clip.setTimestamp(self.grid.getTimestamp())

                # Get the clip region and assign it to the grid
                # region.
                rgn = self.clip.getENFRegion()
                self.__setRegion(rgn)

                # Get the region from the curve - just for diagnostics
                rgn = self.enfAudioCurveRegion.getRegion()
                print(f"__onLoadGridHistoryDone: enfAudioCurveRegion={rgn}")

                # Set x-axis range so that the clip is recognisable
                t = self.clip.getTimestamp()
                self.enfPlot.setXRange(t, t + self.clip.getDuration())

                # Plot all clip-related curves
                print(f"Clip: {self.clip.getTimestamp()}, grid: {self.grid.getTimestamp()}")
                self.__plotGridEnf()()
                self.__plotClipEnf()()
                self.clip.__plotClipSpectrum()
            self.unsetCursor()
            self.__plotGridEnf()
            self.tabs.setCurrentIndex(2)
        else:
            self.unsetCursor()

            # Loading grid data failed
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Information")
            dlg.setIcon(QMessageBox.Information)
            dlg.setText(f"Could not get ENF values")
            dlg.exec()
        self.__setButtonStatus()


    @pyqtSlot(str, int)
    def __gridHistoryLoadingProgress(self, hint, progress):
        """Update text and percentage of the progress dialog. Called when the GUI
        thread receives a 'progress' signal.

        :param hint: Text to be displayed in the progress dialog.
        :param progress: Progress count, incremented for each month of data retrieved
        from either the database or from the internet.
        """
        print("__gridHistoryLoadingProgress:", hint, progress)
        self.__ldGridProgDlg.setLabelText(hint)
        self.__ldGridProgDlg.setValue(progress)
        if self.__ldGridProgDlg.wasCanceled():
            print("Was canceled")


    def __onMatchClicked(self):
        """Called when the 'match' button is clicked.

        The method finds the best match of the ENF series of the clip
        (self.clip) and the ENF series of the chosen grid (self.grid).
        Result of the matching process are the values: (1) The time
        offset in seconds from the beginning of the grid ENF,
        (2) a quality indication, and (3) an array of correlation values.
        """

        if self.enfAudioCurveRegion is not None:
            roi = self.enfAudioCurveRegion.getRegion()
            self.clip.setENFRegion((int(roi[0]), int(roi[1])))

        now = datetime.datetime.now()
        print(f"{now} ... starting")
        algo = self.cb_algo.currentText()
        assert algo in ('Convolution', 'Pearson', 'Euclidian')

        ## Progress dialog
        matchingSteps = self.grid.getMatchingSteps(self.clip)
        print(f"__onMatchClicked: {matchingSteps} steps")
        self.matchingProgDlg = QProgressDialog("Trying to determine time of recording, computing best fit ...",
                                               "Cancel",
                                               0, matchingSteps, self)
        self.matchingProgDlg.setWindowTitle("Matching clip")
        self.matchingProgDlg.setWindowModality(Qt.WindowModal)
        self.matchingProgDlg.canceled.connect(self.grid.onCanceled)

        #
        corr = self.grid.matchClip(self.clip, algo, self.__matchingProgress)
        if corr:
            #self.__showMatches(t, q, corr)
            # Adjust the timestamp of the clip
            t = self.grid.getMatchTimestamp()
            self.clip.setTimestamp(t)

            # Zoom into the matched time range
            r = self.grid.getMatchRange()
            self.enfPlot.setXRange(r[0], r[1], padding=0.5)

            # Plot curves
            self.clip.plotENF()
            self.clip.plotENFsmoothed()
            self.grid.plotCorrelation()
            self.tabs.setCurrentIndex(1)

            # Set text fields
            self.e_offset.setText(str(t))
            ts = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')
            self.e_date.setText(ts)
            self.e_quality.setText(str(self.grid.getMatchQuality()))

            # self.unsetCursor()
            now = datetime.datetime.now()
            print(f"__onMatchClicked: {now} ... done")
            if self.enfAudioCurveRegion is not None:
                # --- Adjust region of interest ---
                print(f"__onMatchClicked: region={self.enfAudioCurveRegion.getRegion()}")
                rgn = self.clip.getENFRegion()
                self.__setRegion(rgn, movable=False)
                # ----------------------------------

        self.__setButtonStatus()


    @pyqtSlot()
    def __onRegionChanged(self):
        """Called when the user has dragged one of the region boundaries.

        Queries the actual region boundaries from the plot widget and
        sets the region in the clip.
        """
        rgn = self.enfAudioCurveRegion.getRegion()
        self.clip.setENFRegion(rgn)


    def __matchingProgress(self, value):
        """Called by matchXxxx method to indicate the matching progress."""
        self.matchingProgDlg.setValue(value)
        #print(f"__matchingProgress: {value}")


    def __plotClipEnf(self):
        t, freq = self.clip.getEnf()
        self.__enfAudioCurve.setData([], [])
        self.__enfAudioCurve.setData(t, freq)

        t, freq = self.clip.getEnfs()
        if freq is not None:
            self.__enfSmoothedAudioCurve.setData([], [])
            self.__enfSmoothedAudioCurve.setData(t, freq)


    def __plotGridEnf(self):
        t, freq = self.grid.getEnf()
        self.__enfGridCurve.setData([], [])
        self.__enfGridCurve.setData(t, freq)


    def __plotClipSpectrum(self):
        """Plot the spectrum of the input signal."""
        self.__clipSpectrumCurve.setData([], [])
        fft_freq, fft_ampl = self.clip.makeSpectrum()
        if fft_freq is not None and fft_ampl is not None:
            self.__clipSpectrumCurve.setData(fft_freq, fft_ampl)


    def __plotClipSpectrogram(self):
        # https://github.com/drammock/spectrogram-tutorial/blob/main/spectrogram.ipynb
        f, t, Sxx = self.clip.makeSpectrogram()

        # Fit the min and max levels of the histogram to the data available
        self.spectogr_hist.setLevels(np.min(Sxx), np.max(Sxx))

        # Sxx contains the amplitude for each pixel
        self.spectorgr_img.setImage(Sxx)

        # Scale the X and Y Axis to time and frequency (standard is pixels)
        tr = QtGui.QTransform()
        xscale = t[-1]/np.size(Sxx, axis=1)
        yscale = f[-1]/np.size(Sxx, axis=0)
        print(f"Scale spectorgram: spectorgr_img shape={Sxx.shape}, xscale={xscale}, yscale={yscale}")
        tr.scale(xscale, yscale)
        self.spectorgr_img.setTransform(tr)

        # Limit panning/zooming to the spectrogram
        self.spectrogr_plot.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])


    def closeEvent(self, event):
        """Called when the main window is closed."""
        self.__saveSettings()
        super().closeEvent(event)


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

        assert(type(settings) == QSettings)

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

        self.e_databasePath.setText(self.settings.value("paths/database"))


    def save(self):
        self.settings.setValue("paths/database", self.e_databasePath.text())
        self.accept()


class FlimmerApp(QApplication):
    """ Create a FlimmerView object and show it. """
    def __init__(self, argv):
        super(FlimmerApp, self).__init__(argv)
        self.view = FlimmerView()

    def show(self):
        self.view.show()


if __name__ == '__main__':
    try:
        app = FlimmerApp([])
        app.show()
        sys.exit(app.exec_())
    except MemoryError as e:
        print(e)
