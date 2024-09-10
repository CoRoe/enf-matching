#
# Collection of classes that deal with the detection and processing of ENF
# (Electirc Network Frequencies) signal in audio and video recordings.
#
# Copyright (C) 2024 conrad.roeber@mailbox.org
#
# The functions are:
#
# - Read an preprocess audio and video recordings. Preprocessing is slightly
#   different for audio and video.
#
# - Extract the ENF component from the original signal.
#
# - Match the extracted ENF component against historical ENF values to
#   determine the recording time.
#
# The base class is Enf; there are derived classes for different types of
# media files and processing algorithms.
#

import wave
import datetime
from scipy import signal, fft
import numpy as np
import pandas as pd
import pyqtgraph as pg
import subprocess
import json
import array as arr
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


def notch_filter(data, f0, fs, quality, filename=None):
    """
    Pass data through a notch filter.

    :param f0: The fundamental frequency of the notch filter (the spacing
    between its peaks).
    :param fs: The sampling frequency of the signal.
    :param quality: The quality of the filter.

    The filter removes the fundamental frequency fs and its multiples.
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iircomb.html
    b, a = signal.iircomb(f0, quality, ftype='notch', fs=fs)
    output_signal = signal.filtfilt(b, a, data).astype(np.int16)
    if filename is not None:
        with open('/tmp/video.csv', 'w') as f:
            for i in range(len(data)):
                f.write(f"{data[i]},{output_signal[i]}\n")
    return output_signal


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

    # TODO: Return a numpy array for performance
    print(f"enf_series: sample freq={fs}, nom. freq={nominal_freq}, freq band={freq_band_size}, harmonic={harmonic_n}")
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
        :param data: Array of input data
        :param max_idx: Index into data
        :param bin_size:
        :returns:

        https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
        """
        left = data[max_idx - 1]
        center = data[max_idx]
        right = data[max_idx + 1]

        p = 0.5 * (left - right) / (left - 2 * center + right)
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

        enf = [f/float(harmonic_n) for f in max_freqs]
        return enf
    else:
        return None


def sel_stream_min_variation(video_streams):
    """From a list of videao streams select the one that has the least variation.

    :param video_streams: The list of viedeo stream.
    """

    assert type(video_streams) == np.ndarray and len(video_streams.shape) == 2

    deltas = [np.max(video_stream) - np.min(video_stream) for video_stream in video_streams]
    return video_streams[np.argmin(deltas)]


def sel_stream_max_energy(streams, fs, fc, df):
    """From a list of streams select the one that has the most energy in a given frequency band.

    :param streams: The list of streams.
    :param fs: Sampling frequency
    :param fc: The frequency where the maximuman energy is expected.
    :param df: The bandith
    """

    assert type(streams) == np.ndarray and len(streams.shape) == 2

    min_idx = round(len(streams[0]) * (fc - df) / fs)
    max_idx = round(len(streams[0]) * (fc + df) / fs)

    per_stream_ampl = np.zeros(0)
    for stream in streams:
        ampl = abs(fft.fft(stream))
        amplc = np.sum(ampl[min_idx:max_idx])
        per_stream_ampl = np.append(per_stream_ampl, amplc)

    best_idx = np.argmax(per_stream_ampl)
    print(f"sel_stream_max_energy({fs}, {fc}, {df}) -> {best_idx}")

    # debug
    ampl = abs(fft.fft(streams[best_idx]))

    return streams[best_idx]


class Enf:
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
        # ENFcurve.setData([], [])
        ENFcurve([], [])


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
            return enf[self.region[0] : self.region[1]]
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
                assert ds_factor * 8000 == self.fs
                self.data = None

                # Read chunks, downsample them
                wav_buf = wav_f.readframes(1000000)
                while len(wav_buf) > 0:
                    # print(len(wav_buf))
                    nw = signal.decimate(
                        np.frombuffer(wav_buf, dtype=np.int16), ds_factor
                    )
                    # print("After decimation:", len(nw))
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

            assert type(self.data) == np.ndarray
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
        assert self.data is not None

        self.nominal_freq = nominal_freq
        self.freq_band_size = freq_band_size
        self.harmonic = harmonic
        enf_output = enf_series(
            self.data, self.fs, nominal_freq, freq_band_size, harmonic_n=harmonic
        )

        # stft is the Short-Term Fourier Transfrom of the audio file, computed
        # per second.
        # self.stft = enf_output['stft']

        # ENF are the ENF values
        # TODO: Use array multiplication
        enf = [int(e * 1000) for e in enf_output]
        self.enf = np.array(enf)
        assert type(self.enf) == np.ndarray


    def plotENF(self):
        """Plot the cureve of ENF values.

        This works for ENF values in both clips and grid data. Note that ENFcurve.setData
        cumulates so the existing data have to be removed.
        """
        assert self.ENFcurve is not None
        timestamps = list(range(self._timestamp, self._timestamp + len(self.enf)))
        self.ENFcurve(timestamps, self.enf)


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
        idxs_outliers = np.nonzero(d > threshold * mdev)[0]
        for i in idxs_outliers:
            if i - win < 0:
                x_corr[i] = np.median(
                    np.append(self.enf[0:i], self.enf[i + 1 : i + win + 1])
                )
            elif i + win + 1 > len(self.enf):
                x_corr[i] = np.median(
                    np.append(self.enf[i - win : i], self.enf[i + 1 : len(self.enf)])
                )
            else:
                x_corr[i] = np.median(
                    np.append(self.enf[i - win : i], self.enf[i + 1 : i + win + 1])
                )
        self.enfs = x_corr


    def clearSmoothedENF(self):
        self.enfs = None


    def plotENFsmoothed(self):
        self.ENFscurve([], [])
        if self.enfs is not None:
            timestamps = list(range(self._timestamp, self._timestamp + len(self.enfs)))
            self.ENFscurve(timestamps, self.enfs)


    def getTimestamp(self):
        """Return the timestamp of the object."""
        return self._timestamp


    def setTimestamp(self, timestamp):
        self._timestamp = timestamp


    def dumpDataToFile(self, fn):
        with open(fn, 'w') as fp:
            for value in self.data:
                fp.write(str(value))
                fp.write('\n')


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

    def __init__(self, databasePath,
                 ENFcurveCallback, correlationCurveCallback):
        super().__init__(ENFcurveCallback)
        self.__correlationCurve = correlationCurveCallback
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
        assert self.databasePath
        assert type(year) == int and year > 1970
        assert type(month) == int and month >= 1 and month <= 12
        assert location != 'Test', "Handled elsewhere"

        data_source = GridDataAccessFactory.getInstance(location, self.databasePath)
        self.enf, self._timestamp = data_source.getEnfSeries(
            year, month, n_months, progressCallback
        )
        assert self.enf is None or type(self.enf) == np.ndarray
        assert type(self._timestamp == int)


    def loadCSVFile(self, fn):
        """Load grid ENF values from a local CSV file.

        @þaram fn: The file to load the data from.
        """
        df = pd.read_csv(fn)
        #print(df)
        enf = df['frequency'].astype(float) * 1000
        self.enf = np.array(enf)
        t = datetime.datetime.fromisoformat(df['time'][0])
        self._timestamp = t.strftime('%s')
        assert type(self.enf) == np.ndarray
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
        assert type(clip) in (AudioClipEnf, VideoClipEnf)
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
        assert algo in ("Pearson", "Euclidian", "Convolution")
        if algo == "Pearson":
            r = self.__matchPearson(clip, progressCallback)
        elif algo == "Euclidian":
            r = self.__matchEuclidianDist(clip, progressCallback)
        elif algo == "Convolution":
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
        assert type(clip) in (AudioClipEnf, VideoClipEnf)

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
        # timestamp = clip.getTimestamp()
        print(f"Len clip_enf: {len(clip_enf)}, len(enf): {len(self.enf)}")

        enf = self.enf
        n_steps = len(enf) - len(clip_enf) + 1

        try:
            corr = [
                np.corrcoef(enf[step : step + len(clip_enf)], clip_enf)[0][1]
                for step in step_enum(n_steps, progressCallback)
                if not canceled()
            ]
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
            # return timestamp + max_index, corr[max_index], corr
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

        assert type(clip) in (AudioClipEnf, VideoClipEnf)

        print(f"Start Euclidian correlation computation: {datetime.datetime.now()} ...")
        # timestamp = clip.getTimestamp()

        enf = self.enf
        clip_enf = clip._getENF()

        n_steps = len(enf) - len(clip_enf) + 1
        progressCallback(0)
        try:
            mse = [
                ((enf[step : step + len(clip_enf)] - clip_enf) ** 2).mean()
                for step in step_enum(n_steps, progressCallback)
                if not canceled()
            ]
            # corr = [spatial.distance.cdist([clip_enf[step:step+len(enf)], enf],
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
            # return timestamp + min_index, corr[min_index], corr
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
        xcorr_norm = xcorr / ref_normalization / signal_normalization / len(enf)

        # Store results in instance variables
        self.corr = xcorr_norm
        self.t_match = timestamp + max_index - len(enf) // 2 - clip.region[0]
        self.quality = xcorr_norm[max_index]

        # Signal that we are done
        progressCallback(n_steps)

        # Always succeeds
        return True
        # return timestamp + max_index - self.clip_len_s//2, xcorr_norm[max_index], xcorr_norm

    def getMatchTimestamp(self):
        return self.t_match


    def getMatchRange(self):
        return self.matchRange


    def getMatchQuality(self):
        return self.quality


    def plotCorrelation(self):
        timestamps = list(range(self._timestamp, self._timestamp + len(self.corr)))
        self.__correlationCurve(timestamps, self.corr)


class AudioClipEnf(Enf):
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
        self.ENFscurve([], [])
        self.spectrumCurve([], [])

        self.enfs = None
        self.fft_freq = None
        self.fft_ampl = None
        self.data = None
        self.aborted = False
        self.region = None
        self._timestamp = 0


    def makeEnf(self, nominal_freq, freq_band_size, harmonic=1):
        """Creates an ENF series from the sample data.

        :param: nominal_freq: Nominal grid frequency in Hz; usually 50 or 60 Hz
        :param: freq_band_size: Size of the frequency band around *nominal_freq* in Hz
        :param: harmonic:

        The method takes self.data (the samples of the audio recording) and
        computes self.enf (the series of frequencies of the 50 or 60 Hz hum of
        the recording.)

        """
        assert self.data is not None

        self.nominal_freq = nominal_freq
        self.freq_band_size = freq_band_size
        self.harmonic = harmonic
        enf_output = enf_series(self.data, self.fs, nominal_freq,
                                freq_band_size,
                                harmonic_n=harmonic)

        # stft is the Short-Term Fourier Transfrom of the audio file, computed
        # per second.
        # self.stft = enf_output['stft']

        # ENF are the ENF values
        # TODO: Use array multiplication
        # if enf_output['enf'] is not None:
        if enf_output is not None:
            enf = [int(e * 1000) for e in enf_output]
            self.enf = np.array(enf)
        else:
            self.enf = None
        assert self.enf is None or type(self.enf) == np.ndarray


    def makeFFT(self):
        """ Compute the spectrum of the original audio or video recording.

        :param: self.data: sample data of the audio or video file
        :param: self.fs: sample frequency
        :returns: Tuple (frequencies, amplitudes)
        """
        # https://docs.scipy.org/doc/scipy/tutorial/fft.html#d-discrete-fourier-transforms
        # Result is complex.
        assert self.data is not None

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
        self.region = (
            int(region[0]) - self._timestamp,
            int(region[1]) - self._timestamp,
        )
        print("setENFRegion:", self.region)


    def getENFRegion(self):
        """It is an error to query the region before it has been set with setENFRegion()."""
        rgn = (self.region[0] + self._timestamp, self.region[1] + self._timestamp)
        return rgn


    def getDuration(self):
        """Length of the clip in seconds."""
        return self.clip_len_s


    def sampleRate(self):
        """Return the sample rate in samples / second."""
        return self.fs


    def plotSpectrum(self):

        assert self.fft_ampl is not None and self.fft_freq is not None
        # self.spectrumCurve.setData([])
        # self.spectrumCurve.setData(self.fft_freq, self.fft_ampl)
        self.spectrumCurve(self.fft_freq, self.fft_ampl)


class VideoClipEnf(Enf):

    method_gridroi = 0
    method_rs = 1

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
        # FIXME: Cannot handle 'webm' format because it codes the duration in a different way.
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
                if stream['codec_type'] == 'video':
                    # We know it is a video stream; set some instance variables
                    self.height = stream['height']
                    self.width = stream['width']
                    self.clip_len_s = int(float(stream['duration']))
                    # Someting like '30000/1001'
                    fr = stream['r_frame_rate'].split('/')
                    #self.frame_rate = int(stream['r_frame_rate'][:-2])
                    self.frame_rate = round(int(fr[0])/int(fr[1]))
                    break
            else:
                return None
            return p
        else:
            return None


    def aliasFreqs(self, gridFreq, gridHarmonic, vidFrameHarmonic):
        """Given the nominal grid frequency, the harmonic of the grid frequency, the
        video frame rate, and the video frame rate harmonic compute the alias frequency.
        """
        assert self.__method in (VideoClipEnf.method_gridroi, VideoClipEnf.method_rs)

        f = gridFreq * gridHarmonic + self.frame_rate * vidFrameHarmonic
        if self.__method == VideoClipEnf.method_gridroi:
            if f > self.frame_rate // 2:
                # Illegal combination
                f = None
        else:
            if f > self.fs // 2:
                f = None
        return f


    def checkAliasFreqs(self, gridFreq, gridHarmonic, vidFrameHarmonic):
        """Determine if the combination of ... is legal.
        The alias frequency must be lower than half the sampling frequency.
        """


    def loadVideoFileRollingShutter(self, filename, readout_time):
        """Read a video file.

        :param filename: The filename; type can be anything that ffmpeg supports.
        :param readout_time: Time [ms] it takes to read out all scan lines from
        the image sensor. This is a camera parameter.
        """

        self.__method = VideoClipEnf.method_rs

        # Frame size in bytes
        frameSize = self.width * self.height

        # Intended sampling frequency; may become a bit higher because of
        # the added interpolated ('extra') scan lines.
        fs = 600

        # Number of scan lines to be interpolated
        extra_scan_lines = self.height * (1 - readout_time/1000 * self.frame_rate)

        # To have an effective sample frequency of say 600 Hz at a frame rate of 30 fps,
        # each frame must be split into 600 / 30 = 20 slices.
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
        frame_number = 0
        while True:
            # Read an entire image frame. Frame format is y800: 1 luminance byte per pixel.
            YframeBuffer = proc.stdout.read(frameSize)
            if len(YframeBuffer) < frameSize:
                print(f"Found {frame_number} frames; residual frame: {len(YframeBuffer)} bytes")
                break
            # Convert to array for easier handling
            YArray = np.frombuffer(YframeBuffer, dtype=np.uint8)
            for s in range(0, frameSize, slice_size * self.width):
                #print(s, s + slice_size * self.width)
                avg = np.uint16(np.average(YArray[s:s + slice_size * self.width]) * 256)
                clipLuminanceArray = np.append(clipLuminanceArray, avg)

            # For each of the 'extra' (interpolated scan lines) add the average
            # of the last 'real' scan line:
            for s in range(extra_slices):
                clipLuminanceArray = np.append(clipLuminanceArray, avg)
            frame_number += 1

        self.data = clipLuminanceArray
        self.fs = round(fs * (n_slices + extra_slices) / n_slices)

        # Region is the entire clip; values are in seconds
        self.region = (0, self.clip_len_s)

        return clipLuminanceArray


    def loadVideoFileGridROI(self, filename, v_slices, h_slices, csv=None):
        """Open a video clip and load its luminance as time series into memory.

        :param filename: Name of the video clip.
        :param v_slices: Number of rectangles stacked vertically.
        :param h_slices: Number of rectangles side-by-side.
        :returns self.data: List of arrays where each element contains a series of luminance
        values.
        :returns self.fs: Scan frequency - the frames-per-second value of the clip.

        Each frame of the video is divided into rectangles; in total there are v_slices * h_slices
        rectangles. For each such rectangle the average luminance is computed and stored.
        There is thus v_slices * h_slices time series.
        """

        self.__method = VideoClipEnf.method_gridroi

        # Frame size in bytes
        frameSize = self.width * self.height

        # Horizontal slice size
        hsize = self.width // h_slices

        # Vertical slice size
        vsize = self.height // v_slices

        #slice_luminance = np.empty(v_slices * h_slices)
        #luminance_per_slice = np.zeros(dtype=np.uint16, shape=(self.nb_frames+1, v_slices, h_slices))
        #luminance_per_slice = [arr.array('H')] * v_slices * h_slices
        luminance_per_slice = [arr.array('H') for v in range(v_slices) for h in range(h_slices)]

        cmd = ['/usr/bin/ffmpeg', '-i', filename,
               '-vf', 'extractplanes=y', '-f', 'rawvideo',
                '-'
                ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
        frame_number = 0
        while True:
            # Read an entire image frame. Frame format is y800: 1 luminance byte per pixel.
            YframeBuffer = proc.stdout.read(frameSize)
            if len(YframeBuffer) < frameSize:
                print(f"Found {frame_number} frames; residual frame: {len(YframeBuffer)} bytes")
                break

            image = np.frombuffer(YframeBuffer, dtype=np.uint8)
            image = np.reshape(image, (self.height, self.width))
            # Indices are (row, column)
            for r in range(v_slices):
                for c in range(h_slices):
                    #print(r, c)
                    rect = image[vsize * r : vsize * r + vsize,
                                 hsize * c : hsize * c + hsize]
                    avg = np.uint16(np.average(rect * 256))
                    luminance_per_slice[h_slices * r + c].append(avg)

            frame_number += 1

        if csv:
            with open(csv, 'w') as fp:
                for frame in range(frame_number):
                    for r in range(v_slices * h_slices):
                        # fp.write(f"{luminance_per_slice[r][frame]},")
                        a = luminance_per_slice[r]
                        fp.write(f"{str(a[frame])},")
                    fp.write('\n')

        print(np.shape(luminance_per_slice))
        self.fs = self.frame_rate
        self.data = [np.array(l) for l in luminance_per_slice]

        # Region is the entire clip; values are in seconds
        self.region = (0, self.clip_len_s)



    def makeEnf(self, grid_freq, nominal_freq: int, grid_harmonic, freq_band_size, vid_harmonic,
                notchf_qual=0):
        """Extract an ENF time series from a video signal.

        :param nominal_freq: The frequency where ENF deviations are expected.
        :param freq_band_size: The width of the bandpass filter around nominal_freq
        :param notchf_qual: The quality of the notch filter that removes the image
        frame frequency and its harmonics.
        :param self.data: Video signal
        :param self.frame_rate: Image frame frequency; typically 24 or 30 frames per second.
        :param self.fs: Sampling frequency

        :returns: True if the ENF time series could be extracted and False if some error
        occurred.
        :returns self.data: 'Best' time series
        :returns self.enf: ENF time series

        Processing depends on the method that has been chosen when loading the video (rolling
        vs. global shutter).
        """

        print(f"makeEnf: method is {self.__method}")
        print(f"makeEnf: grid_freq={grid_freq}, nominal_freq={nominal_freq}, freq band:{freq_band_size}")

        assert self.__method in (VideoClipEnf.method_rs, VideoClipEnf.method_gridroi)
        data = np.array(self.data)

        if self.__method == VideoClipEnf.method_rs:
            # Compute the spectrum of the unprocess input data
            spectrum = fft.fft(data)
            self.fft_freq = fft.fftfreq(len(spectrum), 1.0 / self.fs)
            self.fft_ampl = np.abs(spectrum)

            locut = (grid_freq - freq_band_size) * grid_harmonic
            hicut = (grid_freq + freq_band_size) * grid_harmonic

            # Apply notch filter that removes any signal components of the frame rate
            # and its harmonics.
            print(f"Notch filter: frame rate={self.frame_rate}, sample freq={self.fs}, qual={notchf_qual}")
            data = notch_filter(data, self.frame_rate, self.fs, notchf_qual, "/tmp/video.csv")

            # Apply a band-pass Butterworth filter that leaves only the frequency range
            # where the ENF is expected:
            print(f"Band pass: locut={locut}, hicut={hicut}, sample freq={self.fs}, order=10")
            data = butter_bandpass_filter(data, locut, hicut, self.fs, 10)
            enf = enf_series(data, self.fs, nominal_freq, freq_band_size, harmonic_n=grid_harmonic)

        elif self.__method == VideoClipEnf.method_gridroi:
            locut = nominal_freq - freq_band_size
            hicut = nominal_freq + freq_band_size

            assert len(np.shape(data)) == 2
            #data = sel_stream_min_variation(data)
            data = sel_stream_max_energy(data, self.fs, nominal_freq, freq_band_size)

            # Compute the spectrum of the unprocess input data
            spectrum = fft.fft(data)
            self.fft_freq = fft.fftfreq(len(spectrum), 1.0 / self.fs)
            self.fft_ampl = np.abs(spectrum)

            # Apply a band-pass Butterworth filter that leaves only the frequency range
            # where the ENF is expected:
            print(f"Band pass: locut={locut}, hicut={hicut}, sample freq={self.fs}, order=10")
            data = butter_bandpass_filter(data, locut, hicut, self.fs, 10)
            enf = enf_series(data, self.fs, nominal_freq, freq_band_size, harmonic_n=1)

        if enf is not None:
            # Convert into np.array for uniformness
            # enf = (nominal_freq - enf) / vid_harmonic + nominal_freq
            self.enf = (nominal_freq - np.array(enf)) + grid_freq
        else:
            self.enf = None

        # Spectrum must be two1-dimensional arrays
        assert type(self.fft_freq) == np.ndarray and len(np.shape(self.fft_freq)) == 1
        assert type(self.fft_ampl) == np.ndarray and len(np.shape(self.fft_ampl)) == 1

        assert self.enf is None or (type(self.enf) == np.ndarray and len(np.shape(self.enf)) == 1)
        return self.enf is not None


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
