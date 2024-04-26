#
# Generates a video file with an ENF pattern. The file is always 120 seconds
# long. Format is always 1080x1920.
#

import numpy as np
import subprocess
import array as arr
import argparse
import matplotlib.pyplot as plt

#
# Parameters
#

duration = 120              # Duration in seconds


def gen_signals(fs: int, fc: int, duration: int):
    """Generates an ENF signal at the nominal grid frequency that is slightly
    frequency modulated.

    :param fc: Nominal grid frequency
    :param fs: Sampling frequency; must be an integer multiple of the video
    frame rate
    :param duration: Duration of the generated video
    :returns t: Time series, spacing is the sampling frequency, i.e. some
    integer
    multiple of the frame rate
    :returns sig_mod: Frequency modulated signal; absolute value of a sine wave.
    :returns delta_phi:
    """

    # Time axis
    t = np.arange(0, duration, 1/fs)

    # Format: Duration (seconds), frequency deviation factor
    freq_deltas = (
        (20, +0.002),
        (20, -0.002),
        (30,  0.000),
        (30, -0.003),
        (20, +0.003)
    )

    delta_phi = np.empty(0)
    for dt, df in freq_deltas:
        # print(dt, df)
        s = np.ones(dt * fs) * df
        delta_phi = np.append(delta_phi, s)

    #%% modulation
    cum_phi = (np.cumsum(delta_phi) - delta_phi[0])
    # print(int(cum_phi[fs]))
    phi = 2 * np.pi * fc * (t + cum_phi / fs)

    # sig_mod is the ENF signal including the frequency deltas defined in
    # freq_deltas
    sig_mod = np.abs(np.sin(phi))      # modulated signal (absolute value of sine wave)
    # sig_mod = 0.5 * (np.sin(phi) + 1)              # sine wave

    return t, sig_mod, delta_phi


def gen_sensor_signal(sig_mod, fs, fps):
    """Generate a sequence of sensor output values from the frequency modulated
    illumination.

    :param sig_mod: Series of samples of brightness values.
    :param fs: Sampling frequency
    :param fps: Video frame rate
    :returns sig_sensor: Array with the simulated output of the camera sensor.

    Downsample
    """
    samples_per_frame = fs // fps
    total_samples = len(sig_mod)
    sig_sensor = np.array([np.average(sig_mod[t:t+samples_per_frame])
                           for t
                           in range(0, total_samples, samples_per_frame)])

    # Normalise
    min = np.min(sig_sensor)
    max = np.max(sig_sensor)
    normalised = (sig_sensor - min) / (max - min)
    #min = np.min(normalised)
    #max = np.max(normalised)

    assert np.shape(sig_sensor) == (total_samples // samples_per_frame, )
    t_sensor = np.arange(0, duration, 1/fps)
    return t_sensor, normalised


def gen_raw_video_file(sig_sensor, filename, contrast,
                       width, height, speed='medium'):
    """Pipe a raw video file to STDOUT, using the luminance values in sig_sensor.

    :param sig_sensor: Array with luminance values.
    :param filenam: Name of the file that ffmpeg generates.
    :param contrast: Contrast of the generated video file.

    The pixel format that is fed into ffmpeg is RGB 8:8:8.
    """

    assert contrast >= 0 and contrast < 256

    vidfmt = f'{width}x{height}'

    cmd = ["/usr/bin/ffmpeg", '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s',
           vidfmt, '-r', '30', '-pix_fmt', 'rgb24', '-i', 'pipe:', '-c:v',
           'libx264', '-preset', speed, '-qp', '0', '-y', filename]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if p.returncode is None:
        try:
            for l in sig_sensor:
                frm_array = arr.array('B', [int(l * contrast)]) * height * width * 3
                p.stdin.write(frm_array)
        except BrokenPipeError:
            print("Video generation aborted.")
    else:
        _, errors = p.communicate()
        print(errors)


def plot(fs, fps, t, delta_phi, sig_mod, t_sensor, sig_sensor):
    """Plot the generated signal and its spectrum.

    :param fs: The sampling frequency; depends on grid frequency and camera
    frame rate.
    :param t: Time series
    :param delta_phi: time series of simulated grid frequency deviations.
    :param sig_mode: ??
    :param sig_sensor: Time series of luminosity a camera sensor would see
    """

    # Compute the spectrum; subtract the average to avoid a peak at
    # a frequency of 0.
    lum_spectrum = np.fft.fft(sig_sensor - np.mean(sig_sensor))
    freqs = np.fft.fftfreq(len(sig_sensor), 1 / fps)

    fig, (ax0, ax1, ax2) = plt.subplots(3)

    ax0.plot(t, delta_phi)
    ax0.set_title("Frequency deviation")
    ax0.set_xlabel("Time (sec)")

    ax1.plot(t_sensor[:11], sig_sensor[:11])
    ax1.plot(t[:200], sig_mod[:200])
    ax1.set_title("Simulated sensor signal / light intensity")
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Amplitude")

    ax2.plot(freqs, lum_spectrum.real)
    ax2.set_title("Spectrum of simulated sensor signal")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
    print()



if __name__ == '__main__':

    description = """Creates a video file with an ENF signal. Some parameters are hard-coded:
    Length 120 seconds, format is 1080x1920.
    """
    parser = argparse.ArgumentParser(
        prog='fm',
        description=description
        )
    parser.add_argument('filename', help="Output video file")
    parser.add_argument('--fps', default=30, help="Frames per second; default is 30 fps", type=int)
    parser.add_argument('--width', default=1920, type=int, help="Width in pixels. Defaults to 1920")
    parser.add_argument('--height', default=1080, type=int, help="Height in pixels. Defaults to 1080.")
    parser.add_argument('--grid', default=50, help="Nominal grid frequency (Hz); default is 50 Hz",
                        choices=[50, 60], type=int)
    parser.add_argument('--contrast', default=255,
                        help="Contrast of the generated video {0..255}; default is 255",
                        type=int)
    parser.add_argument('--plot', help="Plot spectrum instead of generating a file",
                        action='store_true')
    args = parser.parse_args()

    fs = args.fps * 20

    t, sig_mod, delta_phi = gen_signals(fs, args.grid, duration)
    t_sensor, sig_sensor = gen_sensor_signal(sig_mod, fs, args.fps)
    if args.plot:
        plot(fs, args.fps, t, delta_phi, sig_mod, t_sensor, sig_sensor)
    gen_raw_video_file(sig_sensor, args.filename, args.contrast, args.width, args.height)
