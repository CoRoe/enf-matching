#
# Read video file, extract luminance signal, compute FFT.
#

import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from future.builtins import int
from matplotlib.scale import scale_factory


mains_freq = 50

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
    freq, t, Zxx = signal.stft(data, fs, nperseg=nperseg, noverlap=noverlap)
    return freq, t, Zxx


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

    freq, t, Zxx = stft(filtered_data, downsampled_fs)

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

    bin_size = freq[1] - freq[0]

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
            'freq': freq,
            'tablew': t,
            'Zxx': Zxx,
        },
        'vid_signal': [freq/float(harmonic_n) for freq in max_freqs],
    }



# Read piped output: https://stackoverflow.com/questions/18421757/live-output-from-subprocess-command

def checkFormat(filename):
    cmd = ["/usr/bin/ffprobe", filename]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    output, errors = p.communicate()
    print("Output:", output)
    print("Errors:", errors)

    # Check if fps is always a float
    videoprops = re.compile(r'  Stream.+: Video:.+, (\d+)x(\d+),.+ (\d+\.\d+) fps,')
    m = videoprops.search(errors)
    if m:
        hres = int(m.group(1))
        vres = int(m.group(2))
        fps = float(m.group(3))
        print(f"Mains: {mains_freq} Hz; camera frame rate: {fps}/s; interest: {2*mains_freq - 3*fps} Hz")
        return hres, vres, fps
    else:
        return None, None, None


def readVideoFile(filename, scale_factor, hres):
    """Read a video file.

    :param filename: The filename; type can be anything that ffmpge supports.
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
    assert type(hres) == int

    # Array containing the average brightness of each scan line
    clipLuminanceArray = np.empty((0, ), dtype=np.uint)

    cmd = ['/usr/bin/ffmpeg', '-i', filename,
           '-vf', f'scale=iw/{scale_factor}:-1', '-f', 'rawvideo',
           '-pix_fmt', 'yuyv422', '-'
           ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None)
    while True:
        scanlineYUV = proc.stdout.read(2 * hres)
        if len(scanlineYUV) == 0:
            break
        YUVdataArray = np.frombuffer(scanlineYUV, dtype=np.uint8)
        deinterleavedYUV = [YUVdataArray[idx::2] for idx in range(2)]
        luminanceArray = deinterleavedYUV[0] * 256
        averageLum = np.uint16(np.average(luminanceArray))
        #print(f"Line {len(clipLuminanceArray)}: {YUVdataArray[0:4]}..{YUVdataArray[60:64]}..{YUVdataArray[480:484]}")
        clipLuminanceArray = np.append(clipLuminanceArray, averageLum)

    return clipLuminanceArray


def compute_spectrum(data, sr):
    """Compute the spectrum of a sequence of values.

    :param data: Sequence of luminence values.
    :param sr: Sample rate in frames per seconds.
    """
    data = data - np.average(data)
    X = np.fft.fft(data)
    Xabs = np.abs(X)        # Spectrum
    N = len(X)              # Number of coefficients
    n = np.arange(N)
    T = N/sr
    freq = n/T              # Array of frequencies
    return freq, Xabs


def show_spectrum(freq, vid_signal):
    #plt.figure(figsize = (12, 6))
    #plt.subplot(121)

    plt.stem(freq, vid_signal, 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(48, 52)
    #plt.ylim(0, 2000)
    plt.show()


if __name__ == '__main__':
    filename = 'philips-hue.mp4'
    hres, vres, fps = checkFormat(filename)

    assert hres == 1920

    # Downscale by 4 so that we have 1080 / 4 == 270 scan lines, 480x270
    scale_factor = 4
    vid_signal = readVideoFile(filename, scale_factor, hres // scale_factor)
    sample_freq = int(fps) * (vres // scale_factor)
    #freq, vres = compute_spectrum(vid_signal, sample_freq)
    #show_spectrum(freq, vres)
    filtered_vid_signal = butter_bandpass_filter(vid_signal, 49, 51, sample_freq, 10)
    freq, vres = compute_spectrum(filtered_vid_signal, sample_freq)
    show_spectrum(freq, vres)

    print(hres, vres, fps)
