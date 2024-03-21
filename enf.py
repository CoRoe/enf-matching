import wave
import datetime
from scipy import signal, fft
import numpy as np
import pandas as pd
import pyqtgraph as pg
import subprocess
import json
from griddata import GridDataAccessFactory


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
    This is the same return format as scipy's stft function. Returns None if STFT
    throws a ValueError exception.
    """
    window_size_seconds = 16
    nperseg = fs * window_size_seconds
    noverlap = fs * (window_size_seconds - 1)
    # STFT will throw an except if the data series is too short
    try:
        f, t, Zxx = signal.stft(data, fs, nperseg=nperseg, noverlap=noverlap)
    except ValueError:
        f, t, Zxx = None, None, None
    return f, t, Zxx


def enf_series(data, fs, nominal_freq, freq_band_size, harmonic_n=1):
    """Extracts a series of ENF values from `data`, one per second.

    :param data: list of signal sample amplitudes
    :param fs: the sample rate
    :param nominal_freq: the nominal ENF (in Hz) to look near
    :param freq_band_size: the size of the band around the nominal value in which to look for the ENF
    :param harmonic_n: the harmonic number to look for
    :returns: a list of ENF values, one per second or None on error
    """
    print(f"enf_series: sample freq={fs}, grid freq={nominal_freq}, freq band={freq_band_size}, harmonic={harmonic_n}")
    # downsampled_data, downsampled_fs = downsample(data, fs, 300)
    downsampled_data, downsampled_fs = (data, fs)

    locut = harmonic_n * (nominal_freq - freq_band_size)
    hicut = harmonic_n * (nominal_freq + freq_band_size)

    print(f"Band pass: locut={locut}, hicut={hicut}, sample freq={downsampled_fs}, order=10")
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

    if Zxx is not None:
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
        'enf': [f/float(harmonic_n) for f in max_freqs] if Zxx is not None else None,
    }


class Enf():
    """Abstract base class for Electric Network Frequency data.

    Essentially a container for a time series of frequency values (ENF)
    and a _timestamp.

    Clear the curve by setting empty data.
    """
    def __init__(self, ENFcurve):
        self.enf = None
        self.enfs = None
        self._timestamp = None
        self.ENFcurve = ENFcurve
        ENFcurve.setData([], [])


    def _getENF(self, smoothedPreferred=True, onlyRegion=True):
        """Get the ENF time series.

        :param smoothedPreferred: If True, the smoothed version is handed out
        if it exists.
        :param onlyRegion: If true then only the ENF data within the region
        :returns: array with time series of ENF values.
        """
        if smoothedPreferred and self.enfs is not None:
            enf = self.enfs
        else:
            enf = self.enf
        if onlyRegion:
            return enf[self.region[0]:self.region[1]]
        else:
            return enf


    def ENFavailable(self):
        """Check if ENF value are available. The smoothed data may
        or may not be availble."""
        return self.enf is not None


    def loadWaveFile(self, fpath):
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

        self._timestamp = 0

        # Set the region to the whole clip
        self.region = (0, self.clip_len_s)


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
        if enf_output['enf'] is not None:
            enf = [int(e * 1000) for e in enf_output['enf']]
            self.enf = np.array(enf)
        else:
            self.enf = None
        assert self.enf is None or type(self.enf) == np.ndarray


    def plotENF(self):
        """Plot the cureve of ENF values.

        This works for ENF values in both clips and grid data. Note that ENFcurve.setData
        cumulates so the existing data have to be removed.
        """
        assert self.ENFcurve is not None
        self.ENFcurve.setData([], [])
        timestamps = list(range(self._timestamp, self._timestamp + len(self.enf)))
        self.ENFcurve.setData(timestamps, self.enf)


    def getTimestamp(self):
        """Return the timestamp of the object."""
        return self._timestamp


    def setTimestamp(self, timestamp):
        self._timestamp = timestamp


class GridEnf(Enf):
    """Models grid frequency values.

    he ENF series is cached in an sqlite database;
    the path of this database must be passed a parameter to the constructor,
    It may also be changed later. (In this case, all data of the database is
    lost.)

    All time series are internally handled as Numpy arrays with the shape (N,
    0) where N is the number of items of the array. The _timestamp of the first
    element of a time series is kept in an instance variable; the following
    elements are evenly spaced by 1 second.
"""
    def __init__(self, databasePath, ENFcurve, correlationCurve):
        assert type(ENFcurve) == pg.PlotDataItem
        super().__init__(ENFcurve)
        self.correlationCurve = correlationCurve
        self.databasePath = databasePath
        self.t_match = None


    def setDatabasePath(self, path):
        self.databasePath = path


    def loadGridEnf(self, location, year: int, month: int, n_months,
                    progressCallback):
        """Load the grid ENF values from a database.

        :param location: The name/location of the grid
        :param year: The year
        :param month: The number of the month (1 = Jan, 2 = Feb, ...)
        :param n_months: Number of months to get grid data for
        """
        assert(self.databasePath)
        assert(type(year) == int and year > 1970)
        assert(type(month) == int and month >= 1 and month <= 12)
        assert location != 'Test', "Handled elsewhere"

        data_source = GridDataAccessFactory.getInstance(location,
                                                        self.databasePath)
        self.enf, self._timestamp = data_source.getEnfSeries(year, month, n_months,
                                                            progressCallback)
        assert self.enf is None or type(self.enf) == np.ndarray
        assert type(self._timestamp == int)


    def onCanceled(self):
        """Handles the 'cancel' signal from a QProgressDialog.

        Sets the instance variable aborted. Lengthy operations check this
        flag and stop when it is set.
        """
        self.aborted = True


    def getMatchingSteps(self, clip):
        """Return the number of number of iterations. Usefull for a progress
        indicator."""
        assert type(clip) == ClipEnf
        n_steps = len(self.enf) - len(clip._getENF()) + 1
        return n_steps


    def matchClip(self, clip, algo, progressCallback):
        """Compute the time lag where the clip data best matches the grid data.

        :param clip: The clip to be matched.
        :param algo: The matching algorithm to use.
        :param progressCallback: Function to be invoked to signal progreee
        to the caller.

        :returns: True if the function terminated normally or False if the
        computing was cancelled.
        """
        assert algo in ('Pearson', 'Euclidian', 'Convolution')
        if algo == 'Pearson':
            r = self.__matchPearson(clip, progressCallback)
        elif algo == 'Euclidian':
            r = self.__matchEuclidianDist(clip, progressCallback)
        elif algo == 'Convolution':
            r = self.__matchConv(clip, progressCallback)
        else:
            r = False
        if r:
            self.matchRange = (self.t_match, self.t_match + clip.getDuration())
        return r


    def __matchPearson(self, clip, progressCallback):
        """Given a reference clip with ENF values find the best fit with the
        own ENF values.

        :param clip: The ENF series of the grid
        :param progressCallback: Function to be called once in a while
        to signal the progress of the processing.
        :returns: The index into the reference series of thebets match; the
        correlation at that index, an array of correlations for all possible
        indices.

        No -- the _timestamp of the best match.

        The method computes the Pearson correlation between the ENF values in
        the clip and the grid.

        See: https://realpython.com/numpy-scipy-pandas-correlation-python/
        https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        """
        assert(type(clip) == ClipEnf)

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
        clip_enf = clip._getENF()
        #timestamp = clip.getTimestamp()
        print(f"Len clip_enf: {len(clip_enf)}, len(enf): {len(self.enf)}")

        enf = self.enf
        n_steps = len(enf) - len(clip_enf) + 1

        try:
            corr = [np.corrcoef(enf[step:step+len(clip_enf)], clip_enf)[0][1]
                    for step in step_enum(n_steps, progressCallback)
                    if not canceled()]
        except StopIteration:
            print("Cancelled...")
        if self.aborted:
            return False
        else:
            max_index = np.argmax(corr)
            print(f"End Pearson correlation computation {datetime.datetime.now()} ...")
            # Set t_match to the beginning of the clip (not to the region where
            # that was used for matching)
            self.t_match = self._timestamp + max_index - clip.region[0]
            self.quality = corr[max_index]
            self.corr = corr
            #return timestamp + max_index, corr[max_index], corr
            progressCallback(n_steps)
            return True


    def __matchEuclidianDist(self, clip, progressCallback):
        """Given a reference clip with ENF values find the best fit with the
        own ENF values.

        :param clip: The ENF series of the grid
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

        assert(type(clip) == ClipEnf)

        print(f"Start Euclidian correlation computation: {datetime.datetime.now()} ...")
        #timestamp = clip.getTimestamp()

        enf = self.enf
        clip_enf = clip._getENF()

        n_steps = len(enf) - len(clip_enf) + 1
        progressCallback(0)
        try:
            mse = [((enf[step:step+len(clip_enf)] - clip_enf) ** 2).mean()
                    for step in step_enum(n_steps, progressCallback)
                    if not canceled()]
            #corr = [spatial.distance.cdist([clip_enf[step:step+len(enf)], enf],
            #                               [clip_enf[step:step+len(enf)], enf],
            #                               'sqeuclidean')[0][1] for step in step_enum(n_steps, progressCallback)
            #                                if not canceled()]
        except StopIteration:
            print("...canceled")
        if self.aborted:
            return False
        else:
            # Normalise
            corr = mse / np.sqrt(len(mse))
            min_index = np.argmin(corr)
            print(f"End Euclidian correlation computation {datetime.datetime.now()} ...")
            progressCallback(n_steps)
            #return timestamp + min_index, corr[min_index], corr
            self.t_match = self._timestamp + min_index - clip.region[0]
            self.quality = corr[min_index]
            self.corr = corr
            return True


    def __matchConv(self, clip, progressCallback):
        """Compute correlation between clip ENF and grid ENF.

        :param clip: A GridEnf object representing the grid frequencies.
        :param progressCallback: Function to be called to signal progress.
        Used for a progress bar as feedback to the user.
        """
        print("__matchConv")
        grid_freqs = self.enf

        # Get the region of interest
        enf = clip._getENF()
        n_steps = len(grid_freqs) - len(enf) + 1
        timestamp = self._timestamp
        progressCallback(0)
        xcorr = signal.correlate(
            grid_freqs-np.mean(grid_freqs),
            enf-np.mean(enf),
            mode='same')
        max_index = np.argmax(xcorr)
        ref_normalization = pd.Series(grid_freqs).rolling(len(enf),
                                                          center=True).std()
        signal_normalization = np.std(enf)
        xcorr_norm = xcorr/ref_normalization/signal_normalization/len(enf)

        # Store results in instance variables
        self.corr = xcorr_norm
        self.t_match = timestamp + max_index - len(enf)//2 - clip.region[0]
        self.quality = xcorr_norm[max_index]

        # Signal that we are done
        progressCallback(n_steps)

        # Always succeeds
        return True
        #return timestamp + max_index - self.clip_len_s//2, xcorr_norm[max_index], xcorr_norm


    def getMatchTimestamp(self):
        return self.t_match


    def getMatchRange(self):
        return self.matchRange


    def getMatchQuality(self):
        return self.quality


    def plotCorrelation(self):
        self.correlationCurve.setData([], [])
        timestamps = list(range(self._timestamp, self._timestamp + len(self.corr)))
        self.correlationCurve.setData(timestamps, self.corr)


class ClipEnf(Enf):
    """Handle ENF (Electrical Network Frequency) values found in an audio clip.

    Contains methods to match its ENF series against the grid's ENF series.
    As the matching process can be length, it also contains some mechinsm
    to cancel the matching process.
    """
    def __init__(self, ENFcurve, ENFscurve, spectrumCurve):
        super().__init__(ENFcurve)
        self.ENFscurve = ENFscurve
        self.spectrumCurve = spectrumCurve

        # The curves may pre-exist; clear them
        self.ENFscurve.setData([], [])
        self.spectrumCurve.setData([], [])

        self.enfs = None
        self.fft_freq = None
        self.fft_ampl = None
        self.data = None
        self.aborted = False
        self.region = None
        self._timestamp = 0


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


    def fileLoaded(self):
        """Check if a file has been loaded and its PCM data are vailebale."""
        return self.data is not None


    def setENFRegion(self, region: tuple):
        """Set a region of interest for the ENF values. Only ENF values inside
        this region will be used during the matching process.

        :param region: Tuple (lower limit, upper limit). Both values are
        timestamps as seen by the plot widget.

        """
        self.region = (int(region[0]) - self._timestamp,
                       int(region[1]) - self._timestamp)
        print("setENFRegion:", self.region)


    def getENFRegion(self):
        """It is an error to query the region before it has been set with setENFRegion()."""
        rgn = (self.region[0] + self._timestamp,
               self.region[1] + self._timestamp)
        return rgn


    def getDuration(self):
        """ Length of the clip in seconds."""
        return self.clip_len_s


    def sampleRate(self):
        """ Return the sample rate in samples / second. """
        return self.fs


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


    def clearSmoothedENF(self):
        self.enfs = None


    def plotENFsmoothed(self):
        self.ENFscurve.setData([], [])
        if self.enfs is not None:
            timestamps = list(range(self._timestamp, self._timestamp + len(self.enfs)))
            self.ENFscurve.setData(timestamps, self.enfs)


    def plotSpectrum(self):

        assert self.fft_ampl is not None and self.fft_freq is not None
        self.spectrumCurve.setData([])
        self.spectrumCurve.setData(self.fft_freq, self.fft_ampl)


class VideoEnf(Enf):
    def __init__(self, ENFcurve, ENFscurve, spectrumCurve):
        super().__init__(ENFcurve)
        self.ENFscurve = ENFscurve
        self.spectrumCurve = spectrumCurve

        # The curves may pre-exist; clear them
        self.ENFscurve.setData([], [])
        self.spectrumCurve.setData([], [])

        self.enfs = None
        self.fft_freq = None
        self.fft_ampl = None
        self.data = None
        self.aborted = False
        self.region = None
        self._timestamp = 0


    def getVideoProperties(self, filename):
        # -v quiet -show_streams -show_format -print_format json
        cmd = ["/usr/bin/ffprobe", '-v', 'quiet', '-show_streams', '-show_format',
                '-print_format', 'json', filename]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        output, errors = p.communicate()
        if p.returncode == 0:
            #print("Output:", output)
            p = json.loads(output)
            for stream in p['streams']:
                if 'pix_fmt' in stream:
                    # We know it is a video stream; set some instance variables
                    self.height = stream['height']
                    self.width = stream['width']
                    self.clip_len_s = int(float(stream['duration']))
                    # Someting like '30/1'
                    self.frame_rate = int(stream['r_frame_rate'][:-2])
                    break
            else:
                return None
            return p
        else:
            return None


    def loadVideoFile(self, filename, readout_time):
        """Read a video file.

        :param filename: The filename; type can be anything that ffmpeg supports.
        :param readout_time: Time [ms] it takes to read out all scan lines from
        the image sensor. This is a camera parameter.
        """

        # Frame size in bytes
        frameSize = self.width * self.height

        # Intended sampling frequency; may become a bit higher because of
        # the added interpolated ('extra') scan lines.
        fs = 600

        # Number of scan lines to be interpolated
        extra_scan_lines = self.height * (1 - readout_time/1000 * self.frame_rate)

        # To have an effective sample frequency of say 600 Hz at a frame rate of 30 fps,
        # each frame must be split into 600 / 30 = 20 slices.
        # TODO: Better computation
        n_slices = int(fs / self.frame_rate)

        # Number of scan lines per slice
        slice_size = int(self.height / n_slices)

        # Number of slices with interpolated scan lines
        extra_slices = round(extra_scan_lines / slice_size)

        # Array containing the average brightness of each s
        clipLuminanceArray = np.empty((0, ), dtype=np.uint16)

        cmd = ['/usr/bin/ffmpeg', '-i', filename,
               '-vf', 'extractplanes=y', '-f', 'rawvideo',
                '-'
                ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
        while True:
            # Read an entire image frame. Frame format is y800: 1 luminance byte per pixel.
            YframeBuffer = proc.stdout.read(frameSize)
            if len(YframeBuffer) < frameSize:
                break
            # Convert to array for easier handling
            YArray = np.frombuffer(YframeBuffer, dtype=np.uint8)
            for s in range(0, frameSize, slice_size * self.width):
                #print(s, s + slice_size * self.width)
                avg = np.uint16(np.average(YArray[s:s + slice_size * self.width]) * 256)
                clipLuminanceArray = np.append(clipLuminanceArray, avg)

            # For each of the 'extra' (interpolated scan lines) add the average
            # of the last 'real' scan line:
            for s in range(n_slices):
                np.append(clipLuminanceArray, avg)

        self.data = clipLuminanceArray
        self.fs = round(fs * (n_slices + extra_slices) / n_slices)

        # Region is the entire clip; values are in seconds
        self.region = (0, self.clip_len_s)

        return clipLuminanceArray


    def loadVideoFile1_unused(self, filename, scale_factor):
        """Read a video file.

        :param filename: The filename; type can be anything that ffmpeg supports.
        :param scale_factor:
        :param hres: Number of pixels per scan line of the original file.
        :returns luminanceArray: Series of luminance values, averaged over
        either frames or scan lines. The lenght of the array should be a
        multiple of frmaes or scan lines, resp.

        The function passes the input file thorugh ffmpeg to convert it
        to raw grayscale video. One scan line corresponds to one datapoint.
        Pixel format is yuyv422:

        Y0 U Y1 V

        On exit, the region is set to cover the whole video clip.
        """

        assert type(scale_factor) == int
        assert type(self.width) == int
        frameSize = 2 * self.width // scale_factor * self.height // scale_factor

        # Array containing the average brightness of each scan line
        clipLuminanceArray = np.empty((0, ), dtype=np.uint)

        cmd = ['/usr/bin/ffmpeg', '-i', filename,
               '-vf', f'scale=iw/{scale_factor}:-1', '-f', 'rawvideo',
               '-pix_fmt', 'yuyv422', '-'
               ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
        while True:
            # Read an entire image frame
            YUVframeBuffer = proc.stdout.read(frameSize)
            if len(YUVframeBuffer) < frameSize:
                break
            YUVdataArray = np.frombuffer(YUVframeBuffer, dtype=np.uint8)
            deinterleavedYUV = [YUVdataArray[idx::2] for idx in range(2)]
            luminanceArray = deinterleavedYUV[0] * 256

            ar = np.reshape(luminanceArray, (self.height // scale_factor, self.width // scale_factor))
            m = np.array([np.uint16(np.average(line)) for line in ar])
            clipLuminanceArray = np.append(clipLuminanceArray, m)

        self.data = clipLuminanceArray

        # Sampling frequency
        self.fs = self.height / scale_factor * self.frame_rate

        # Region is the entire clip; values are in seconds
        self.region = (0, self.clip_len_s)

        return clipLuminanceArray


    def loadVideoFile_unused(self, filename, scale_factor=4):
        """Read a video file.

        :param filename: The filename; type can be anything that ffmpeg supports.
        :param scale_factor:
        :param hres: Number of pixels per scan line of the original file.
        :returns luminanceArray: Series of luminance values, averaged over
        either frames or scan lines. The lenght of the array should be a
        multiple of frmaes or scan lines, resp.

        The function passes the input file thorugh ffmpeg to convert it
        to raw grayscale video. One scan line corresponds to one datapoint.
        Pixel format is yuyv422:

        Y0 U Y1 V
        """

        assert type(scale_factor) == int
        assert type(self.width) == int

        # Array containing the average brightness of each scan line
        clipLuminanceArray = np.empty((0, ), dtype=np.uint)

        cmd = ['/usr/bin/ffmpeg', '-i', filename,
               '-vf', f'scale=iw/{scale_factor}:-1', '-f', 'rawvideo',
               '-pix_fmt', 'yuyv422', '-'
               ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
        while True:
            scanlineYUV = proc.stdout.read(2 * self.width // scale_factor)
            if len(scanlineYUV) == 0:
                break
            YUVdataArray = np.frombuffer(scanlineYUV, dtype=np.uint8)
            deinterleavedYUV = [YUVdataArray[idx::2] for idx in range(2)]
            luminanceArray = deinterleavedYUV[0] * 256
            averageLum = np.uint16(np.average(luminanceArray))
            #print(f"Line {len(clipLuminanceArray)}: {YUVdataArray[0:4]}..{YUVdataArray[60:64]}..{YUVdataArray[480:484]}")
            clipLuminanceArray = np.append(clipLuminanceArray, averageLum)

        self.data = clipLuminanceArray
        self.fs = self.height / scale_factor * self.frame_rate
        return clipLuminanceArray


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


    def plotENFsmoothed(self):
        self.ENFscurve.setData([], [])
        if self.enfs is not None:
            timestamps = list(range(self._timestamp, self._timestamp + len(self.enfs)))
            self.ENFscurve.setData(timestamps, self.enfs)


    def plotSpectrum(self):

        assert self.fft_ampl is not None and self.fft_freq is not None
        self.spectrumCurve.setData([])
        self.spectrumCurve.setData(self.fft_freq, self.fft_ampl)


    def getDuration(self):
        return self.clip_len_s


    def setENFRegion(self, region: tuple):
        """Set a region of interest for the ENF values. Only ENF values inside
        this region will be used during the matching process.

        :param region: Tuple (lower limit, upper limit). Both values are
        timestamps as seen by the plot widget.

        """
        self.region = (int(region[0]) - self._timestamp,
                       int(region[1]) - self._timestamp)
        print("setENFRegion:", self.region)


    def getENFRegion(self):
        """It is an error to query the region before it has been set with setENFRegion()."""
        rgn = (self.region[0] + self._timestamp,
               self.region[1] + self._timestamp)
        return rgn


    def getFrameRate(self):
        return self.frame_rate

    def getVideoFormat(self):
        return f"{self.width}x{self.height}"

    def fileLoaded(self):
        return self.data is not None

    def clearSmoothedENF(self):
        self.enfs = None
