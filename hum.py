##!/usr/bin/python3

# https://www.pythonguis.com/tutorials/plotting-pyqtgraph/'

import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget,
    QMainWindow,
    QApplication,
    QVBoxLayout,
    QLineEdit,
    QFileDialog,
    QLabel,
    QPushButton,
    QGroupBox,
    QGridLayout,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QTabWidget,
    QDoubleSpinBox,
    QMenuBar,
    QAction,
    QDialog,
    QMessageBox,
    QDialogButtonBox,
    QProgressDialog,
)
from PyQt5.Qt import Qt
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5 import QtGui

import datetime
import os
import subprocess
import json
import numpy as np
from griddata import GridDataAccessFactory
from enf import AudioClipEnf, GridEnf


class HumView(QMainWindow):
    """Display ENF analysis and result display.

    | Action            | x range   |                                 |
    |-------------------+-----------+---------------------------------|
    | Load clip clicked | unchanged |                                 |
    | Analyse clicked   | Clip data |                                 |
    | Load grid clicked | Grid data |                                 |
    | Match clicked     | Clip data | Clip has already been relocated |
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

        def __init__(self, grid, location, year, month, n_months, progressCallback):
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
            self.grid.loadGridEnf(
                self.location, self.year, self.month, self.n_months, self.__on_progress
            )
            self.finished.emit()

        def __on_progress(self, hint, progr):
            """Send a 'progress' signal. The method is called from the underlying
            loadGridEnf() method."""
            self.progress.emit(hint, progr)


    month_names = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

    def __init__(self):
        """Initialize variables and create widgets and menu."""
        super().__init__()
        self.clip = None
        self.grid = None
        self.settings = Settings()
        self.databasePath = self.settings.databasePath()

        self.__enfAudioCurve = None     # ENF series of loaded audio file
        self.__enfSmoothedAudioCurve = None
        self.enfAudioCurveRegion = None
        self.__clipSpectrumCurve = None # Fourier transform of loaded audio file
        self.__enfGridCurve = None      # ENF series of grid
        self.__correlationCurve = None  # Correlation of ENF series of audio
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
        self.spectrogr_img = pg.ImageItem(axisOrder='row-major')
        self.spectrogr_plot.addItem(self.spectrogr_img)

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
        self.spectogr_hist.setImageItem(self.spectrogr_img)

        # If you don't add the histogram to the window, it stays invisible, but I find it useful.
        self.spectrogr_container.addItem(self.spectogr_hist)

        # Add labels to the axis
        self.spectrogr_plot.setLabel('bottom', "Time", units='s')

        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in this case kHz)
        self.spectrogr_plot.setLabel('left', "Frequency", units='Hz')
        self.tabs.addTab(self.spectrogr_container, "Clip Spectrogram")

        #
        # Create a plot widget for the audio clip spectrum and add it to the
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
            name="WAV file spectrum", pen=HumView.spectrumCurveColour
        )
        self.tabs.addTab(self.clipSpectrumPlot, "Clip Spectrum")

        #
        # Create a plot widget for the various ENF curves and add it to the
        # tab
        self.enfPlot = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.enfPlot.setLabel("left", "Frequency (mHz)")
        self.enfPlot.setLabel("bottom", "Date and time")
        self.enfPlot.addLegend()
        self.enfPlot.setBackground("w")
        self.enfPlot.showGrid(x=True, y=True)
        self.enfPlot.plotItem.setMouseEnabled(y=False)  # Only allow zoom in X-axis
        self.__enfAudioCurve = self.enfPlot.plot(
            name="Clip ENF values", pen=HumView.ENFvalueColour
        )
        self.__enfSmoothedAudioCurve = self.enfPlot.plot(
            name="Smoothed clio ENF values", pen=HumView.ENFsmoothedValueColour
        )
        self.__enfGridCurve = self.enfPlot.plot(
            name="Grid frequency history", pen=HumView.GridCurveColour
        )
        self.tabs.addTab(self.enfPlot, "ENF Series")

        # Plots the correlation versus time offset
        self.correlationPlot = pg.PlotWidget(axisItems={"bottom": pg.DateAxisItem()})
        self.correlationPlot.setLabel("left", "correlation")
        self.correlationPlot.setLabel("bottom", "Date / time")
        self.correlationPlot.addLegend()
        self.correlationPlot.setBackground("w")
        self.correlationPlot.showGrid(x=True, y=True)
        self.correlationPlot.plotItem.setMouseEnabled(
            y=False
        )  # Only allow zoom in X-axis
        self.__correlationCurve = self.correlationPlot.plot(
            name="Correlation", pen=HumView.correlationCurveColour
        )
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
        self.b_nominal_freq.setToolTip(
            "The nominal frequency of the power grid at the place of the recording;"
            " 50 Hz in most countries."
        )
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
        self.sp_Outlier_Threshold.setToolTip(
            "Factor defining which ENF values shall be considered invalid outliers"
        )
        analyse_area.addWidget(self.sp_Outlier_Threshold, 1, 2)
        analyse_area.addWidget(QLabel("Window"), 1, 3)
        self.sp_window = QSpinBox()
        self.sp_window.setValue(5)
        analyse_area.addWidget(self.sp_window, 1, 4)

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
        self.l_country.addItem("CSV file")
        grid_area.addWidget(self.l_country, 0, 1)

        grid_area.addWidget(QLabel("Year"), 0, 2)
        self.l_year0 = QComboBox(self)
        for y in range(2024, 2000 - 1, -1):
            self.l_year0.addItem(f"{y}")
        grid_area.addWidget(self.l_year0, 0, 3)
        grid_area.addWidget(QLabel("Month"), 0, 4)
        self.l_month0 = QComboBox()
        self.l_month0.addItems(HumView.month_names)
        grid_area.addWidget(self.l_month0, 0, 5)

        grid_area.addWidget(QLabel("Year"), 1, 2)
        self.l_year1 = QComboBox(self)
        for y in range(2024, 2000 - 1, -1):
            self.l_year1.addItem(f"{y}")
        grid_area.addWidget(self.l_year1, 1, 3)
        grid_area.addWidget(QLabel("Month"), 1, 4)
        self.l_month1 = QComboBox()
        self.l_month1.addItems(HumView.month_names)
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
        self.enfAudioCurveRegion = pg.LinearRegionItem(
            values=region, pen=HumView.regionAreaPen, bounds=region
        )
        self.enfAudioCurveRegion.setBrush(HumView.regionAreaBrush)
        self.enfAudioCurveRegion.setHoverBrush(HumView.regionAreaHoverBrush)
        self.enfAudioCurveRegion.sigRegionChangeFinished.connect(self.__onRegionChanged)
        self.enfAudioCurveRegion.setMovable(movable)
        self.enfPlot.addItem(self.enfAudioCurveRegion)


    def __editSettings(self):
        """Menu item; pops up a 'setting' dialog."""
        dlg = SettingsDialog(self.settings)
        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")


    def __checkDateRange(self):
        """Check if 'to' date is later than 'from' date and computes the
        number of months between 'from' and 'to' date.

        :returns year0: The 'from' year
        :returns month0: The 'from'monnth (1..12)
        :returns n_months: The number of months
        """
        year0 = int(self.l_year0.currentText())
        month0 = self.l_month0.currentIndex()
        year1 = int(self.l_year1.currentText())
        month1 = self.l_month1.currentIndex()
        n_months = (year1 * 12 + month1) - (year0 * 12 + month0) + 1
        print(
            f"Get grid frequencies from {year0}-{month0+1:02} to {year1}-{month1+1:02}, {n_months} months"
        )
        return year0, month0 + 1, n_months


    def __showEnfSources(self):
        self.setCursor(Qt.WaitCursor)
        dlg = ShowEnfSourcesDlg(self)
        if dlg.exec():
            print("Success!")
        else:
            print("Cancel!")
        self.unsetCursor()


    def __setButtonStatus(self):
        """Enables or disables buttons depending on the clip status."""
        audioDataLoaded = self.clip is not None and self.clip.fileLoaded()
        audioEnfLoaded = self.clip is not None and self.clip.ENFavailable()
        gridEnfLoaded = self.grid is not None and self.grid.ENFavailable()

        self.b_analyse.setEnabled(audioDataLoaded)
        self.b_match.setEnabled(audioEnfLoaded and gridEnfLoaded)


    @classmethod
    def __convertToWavFile(cls, fn, tmpfn):
        """Convert a multimedia file to a WAV file.

        :param fn: The input file name
        :param tmp fn: Temporary output file in WAV format.
        """
        cmd = [
            "/usr/bin/ffmpeg",
            "-i",
            fn,
            "-ar",
            "4000",
            "-ac",
            "1",
            "-f",
            "wav",
            tmpfn,
        ]
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output, errors = p.communicate()
        print("Output:", output)
        print("Errors:", errors)
        return p.returncode == 0


    def __onOpenFileClicked(self):
        """Button to open a multimedia file clicked."""
        self.setCursor(Qt.WaitCursor)

        dlg = QFileDialog(self)
        dlg.setNameFilters(["Audio (*.wav *.mp3 *.opus *.aac)", "All (*)"])
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)

        if dlg.exec():
            fileNames = dlg.selectedFiles()
            fileName = fileNames[0]
            self.clip = AudioClipEnf()
            if self.clip.loadAudioFile(fileName, fs=8000):
                self.e_fileName.setText(fileName)
                self.e_duration.setText(str(self.clip.getDuration()))
                self.e_sampleRate.setText(str(self.clip.sampleRate()))

                # Clear all clip-related plots and the region
                if self.enfAudioCurveRegion:
                    self.enfPlot.removeItem(self.enfAudioCurveRegion)
                    self.enfAudioCurveRegion = None

                self.__plotClipSpectrogram()
                self.__plotClipSpectrum()
            else:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Data Error")
                dlg.setIcon(QMessageBox.Information)
                dlg.setText(
                    f"Could not handle {fileName}. Maybe it is not a"
                    " video or audio file."
                )
                dlg.exec()

        self.unsetCursor()
        self.__setButtonStatus()


    def __onAnalyseClicked(self):
        """Called when the 'analyse' button is pressed."""
        # Display wait cursor
        self.setCursor(Qt.WaitCursor)

        self.clip.makeEnf(
            int(self.b_nominal_freq.currentText()),
            float(self.b_band_size.value() / 1000),
            int(self.b_harmonic.value()),
        )
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
        self.__plotClipEnf()

        # Display region; initially, it comprises the whole clip
        rgn = self.clip.getENFRegion()
        self.__setRegion(rgn)

        self.unsetCursor()
        self.tabs.setCurrentIndex(2)
        self.__setButtonStatus()


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
        self.grid = GridEnf(
            self.settings.databasePath()
        )

        if location == "Test":
            self.setCursor(Qt.WaitCursor)
            self.grid.loadAudioFile("71000_ref.wav")
            self.grid.makeEnf(
                int(self.b_nominal_freq.currentText()),
                float(self.b_band_size.value() / 1000),
                int(self.b_harmonic.value()),
            )
            self.__plotGridEnf()
            self.tabs.setCurrentIndex(2)
            self.unsetCursor()
            self.__setButtonStatus()
        elif location == "CSV file":
            fileName, _ = QFileDialog.getOpenFileName(self,
                                                      "Open CSV file with grid frequencies", "",
                                                      "All Files (*);;CSV Files (*.csv)")
            if fileName:
                self.setCursor(Qt.WaitCursor)
                self.grid.loadCSVFile(fileName)
                self.unsetCursor()
                self.__setButtonStatus()
        else:
            if n_months < 1:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Error")
                dlg.setIcon(QMessageBox.Information)
                dlg.setText("'To' date must be later than 'from' date")
                dlg.exec()
            elif n_months > 12:
                dlg = QMessageBox(self)
                dlg.setWindowTitle("Error")
                dlg.setIcon(QMessageBox.Information)
                dlg.setText("Limit are 12 months")
                dlg.exec()
            else:
                self.__ldGridProgDlg = QProgressDialog(
                    "Loading ENF data from inrternet", "Cancel", 0, n_months, self
                )
                self.__ldGridProgDlg.setWindowTitle("Getting ENF data")
                self.__ldGridProgDlg.setCancelButtonText(None)
                self.__ldGridProgDlg.setWindowModality(Qt.WindowModal)
                self.__ldGridProgDlg.forceShow()
                self.__ldGridProgDlg.setValue(0)

                # Move to thread
                self.__loadGridEnfThread = QThread()
                self.__loadGridEnfWorker = HumView.GetGridDataWorker(
                    self.grid,
                    location,
                    year,
                    month,
                    n_months,
                    self.__gridHistoryLoadingProgress,
                )
                self.__loadGridEnfWorker.moveToThread(self.__loadGridEnfThread)

                # Connect signale
                self.__loadGridEnfThread.started.connect(self.__loadGridEnfWorker.run)
                self.__loadGridEnfThread.finished.connect(
                    self.__loadGridEnfThread.deleteLater
                )
                self.__loadGridEnfWorker.finished.connect(self.__loadGridEnfThread.quit)
                self.__loadGridEnfWorker.finished.connect(self.__onLoadGridHistoryDone)
                self.__loadGridEnfWorker.progress.connect(
                    self.__gridHistoryLoadingProgress
                )
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
                self.clip.plotENF()
                self.clip.plotENFsmoothed()
                #self.clip.__plotClipSpectrum()
            #self.grid.plotENF()
            self.__plotGridEnf()
            self.unsetCursor()
            self.tabs.setCurrentIndex(2)
        else:
            # Loading grid data failed
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Information")
            dlg.setIcon(QMessageBox.Information)
            dlg.setText("Could not get ENF values")
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
        self.matchingProgDlg = QProgressDialog(
            "Trying to locate audio recording, computing best fit ...",
            "Cancel",
            0,
            matchingSteps,
            self,
        )
        self.matchingProgDlg.setWindowTitle("Matching clip")
        self.matchingProgDlg.setWindowModality(Qt.WindowModal)
        self.matchingProgDlg.canceled.connect(self.grid.onCanceled)

        #
        corr = self.grid.matchClip(self.clip, algo, self.__matchingProgress)
        if corr:
            # self.__showMatches(t, q, corr)
            # Adjust the timestamp of the clip
            t = self.grid.getMatchTimestamp()
            self.clip.setTimestamp(t)

            # Zoom into the matched time range
            r = self.grid.getMatchRange()
            self.enfPlot.setXRange(r[0], r[1], padding=0.5)

            # Plot curves
            # TODO: Etwas von hinten durch die Brust ins Auge. Gleich die Methoden
            # der Klasse aufrufen.
            # self.clip.plotENF()
            self.__plotClipEnf()
            #self.clip.plotENFsmoothed()
            #self.grid.plotCorrelation()
            #self.__plotCorrelationCurve(x, y)
            self.tabs.setCurrentIndex(2)

            # Set text fields
            self.e_offset.setText(str(t))
            ts = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            self.e_date.setText(ts)
            self.e_quality.setText(str(self.grid.getMatchQuality()))

            # self.unsetCursor()
            now = datetime.datetime.now()
            print(f"__onMatchClicked: {now} ... done")
            if self.enfAudioCurveRegion is not None:
                # --- Adjust region of interest ---
                print(
                    f"__onMatchClicked: region={self.enfAudioCurveRegion.getRegion()}"
                )
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
        # print(f"__matchingProgress: {value}")


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


    def __plotCorrelationCurve(self, x, y):
        self.__correlationCurve.setData([], [])
        self.__correlationCurve.setData(x, y)


    def __plotClipSpectrogram(self):
        # https://github.com/drammock/spectrogram-tutorial/blob/main/spectrogram.ipynb
        f, t, Sxx = self.clip.makeSpectrogram()

        # Fit the min and max levels of the histogram to the data available
        self.spectogr_hist.setLevels(np.min(Sxx), np.max(Sxx))

        # Sxx contains the amplitude for each pixel
        self.spectrogr_img.setImage(Sxx)

        # Scale the X and Y Axis to time and frequency (standard is pixels)
        tr = QtGui.QTransform()
        xscale = t[-1]/np.size(Sxx, axis=1)
        yscale = f[-1]/np.size(Sxx, axis=0)
        print(f"Scale spectorgram: spectrogr_img shape={Sxx.shape}, xscale={xscale}, yscale={yscale}")
        tr.scale(xscale, yscale)
        self.spectrogr_img.setTransform(tr)

        # Limit panning/zooming to the spectrogram
        self.spectrogr_plot.setLimits(xMin=0, xMax=t[-1], yMin=0, yMax=f[-1])


    def __plotClipSpectrum(self):
        """Plot the spectrum of the input signal."""
        self.__clipSpectrumCurve.setData([], [])
        fft_freq, fft_ampl = self.clip.makeSpectrum()
        if fft_freq is not None and fft_ampl is not None:
            self.__clipSpectrumCurve.setData(fft_freq, fft_ampl)


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
            self.table_layout.addWidget(QLabel(locations[i]), i + 1, 0)
            self.table_layout.addWidget(QLabel(fromDate), i + 1, 1)
            # self.table_layout.addWidget(QLineEdit(fromDate), i+1, 1)
            self.table_layout.addWidget(QLabel(toDate), i + 1, 2)

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

        assert type(settings) == Settings

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


class Settings:
    """Keep track of settings."""

    template = {"databasepath": "/tmp/hum.sqlite"}

    def __init__(self):
        """Initialise the setting.

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
        """Save the settings to a JSON file.

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
            if item not in self.settings0:
                self.settings0[item] = Settings.template[item]


    def databasePath(self):
        """Get the database path from the settings."""
        return self.settings["databasepath"]


    def setDatabasePath(self, path):
        self.settings["databasepath"] = path


class HumController(QApplication):
    """Create a HumView object and show it."""

    def __init__(self, argv):
        super(HumController, self).__init__(argv)
        self.view = HumView()

    def show(self):
        self.view.show()


if __name__ == "__main__":
    try:
        app = HumController([])
        app.show()
        app.exec()
    except MemoryError as e:
        print(e)
