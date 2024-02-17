import wave
import datetime
from scipy import signal, fft
import numpy as np
import pandas as pd
import pyqtgraph as pg
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


class Enf():
    """Abstract base class for Electric Network Frequency data.

    Essentially a container for a time series of frequency values (ENF)
    and a _timestamp.
    
    Clear the curve by setting empty data. 
    """
    def __init__(self, ENFcurve):
        self.enf = None
        self._timestamp = None
        self.ENFcurve = ENFcurve
        ENFcurve.setData([], [])


    def getENF(self):
        return self.enf


    def getTimestamp(self):
        return self._timestamp


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

            self._timestamp = 0


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
        

    def plotENF(self):
        """Plot the cureve of ENF values.
        
        This works for ENF values in both clips and grid data.
        """
        assert self.ENFcurve is not None
        timestamps = range(self._timestamp, self._timestamp + len(self.enf))
        self.ENFcurve.setData(timestamps, self.enf)



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
    def __init__(self, databasePath, ENFcurve):
        assert type(ENFcurve) == pg.PlotDataItem
        super().__init__(ENFcurve)
        self.databasePath = databasePath


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

    def getData(self):
        return self.data


    def getENFs(self):
        return self.enfs


    def setTimestamp(self, timestamp):
        self._timestamp = timestamp


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


    def getFFT(self):
        """Get the spectrum.

        Returns FFT frequencies and amplitudes.
        """
        return self.fft_freq, self.fft_ampl


    def getDuration(self):
        """ Length of the clip in seconds."""
        return self.clip_len_s


    def sampleRate(self):
        """ Return the sample rate in samples / second. """
        return self.fs


    def onCanceled(self):
        """Handles the 'cancel' signal from a QProgressDialog.

        Sets the instance variable aborted. Lengthy operations check this
        flag and stop when it is set.
        """
        self.aborted = True


    def getMatchingSteps(self, gridModel):
        """Return the number of number of iterations. Usefull for a progress
        indicator."""
        assert type(gridModel) == GridEnf
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

        No -- the _timestamp of the best match.

        The method computes the Pearson correlation between the ENF values in
        the clip and the grid.

        See: https://realpython.com/numpy-scipy-pandas-correlation-python/
        https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
        """
        assert(type(ref) == GridEnf)

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

        assert(type(ref) == GridEnf)

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
        """Compute correlation between clip ENF and grid ENF.

        :param ref: A GridEnf object representing the grid frequencies.
        :param progressCallback: Function to be called to signal progress.
        Used for a progress bar as feedback to the user.
        """
        print("matchConv")
        grid_freqs = ref.getENF()
        # Get the region of interest
        if self.enfs is not None:
            enf = self.enfs
        else:
            enf = self.enf
        if self.region is not None:
            t0 = self.region[0]
            t1 = self.region[1]
            enf = enf[t0:t1]
        else:
            t0 = 0
        n_steps = len(grid_freqs) - len(enf) + 1
        timestamp = ref.getTimestamp()
        progressCallback(0)
        xcorr = signal.correlate(
            grid_freqs-np.mean(grid_freqs),
            enf-np.mean(enf),
            mode='same')
        max_index = np.argmax(xcorr)
        ref_normalization = pd.Series(grid_freqs).rolling(self.clip_len_s,
                                                          center=True).std()
        signal_normalization = np.std(enf)
        xcorr_norm = xcorr/ref_normalization/signal_normalization/self.clip_len_s
        progressCallback(n_steps)
        return timestamp + t0 + max_index - self.clip_len_s//2, xcorr_norm[max_index], xcorr_norm


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
        
        
    def plotENFsmoothed(self):
        if self.enfs is not None:
            timestamps = list(range(self._timestamp), self._timestamp + len(self.enfs))
            self.ENFscurve.setData(timestamps, self.enfs)
        else:
            self.ENFscurve.setData([], [])
            
            
    def plotSpectrum(self):
        assert self.fft_ampl is not None and self.fft_freq is not None
        self.spectrumCurve.setData(self.fft_freq, self.fft_ampl)

