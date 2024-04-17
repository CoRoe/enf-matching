#
# Generates a video file with an ENF pattern. The file is always 120 seconds
# long.
#

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
import array as arr
import argparse

#
# Parameters
#

fps = 30                    # Frames per second
fs = fps * 20               # Sampling frequency
duration = 120              # Duration in seconds
fc = 50                     # Nominal grid (carrier) carrier frequency
height = 1080
width = 1920


def gen_signals():
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
    sig_mod = np.abs(np.sin(phi))      # modulated signal

    return t, sig_mod, delta_phi


def gen_lum_signal(sig_mod):
    """Generate a sequence of luminance values from sig_mod.

    :param sig_mod: Series of samples of brightness values. The sample frequency is
    fs.
    :returns lum: Array with one luminance value per frame.

    Downsample
    """
    samples_per_frame = fs // fps
    total_samples = len(sig_mod)
    lum = [np.average(sig_mod[t:t+samples_per_frame]) for t in range(0, total_samples, samples_per_frame)]

    assert np.shape(lum) == (total_samples // samples_per_frame, )
    return lum


def gen_raw_video_file(lum, filename):
    """Pipe a raw video file to STDOUT, using the luminance values in lum.

    :param lum: Array with luminance values.

    The output pixedl format is RGB 8:8:8.
    """

    cmd = ["/usr/bin/ffmpeg", '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '1920x1080', '-r',
           '30', '-pix_fmt', 'rgb24', '-i', 'pipe:', '-c:v', 'libx264', '-preset', 'ultrafast',
            '-qp', '0', '-y', filename]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if p.returncode is None:
        for l in lum:
            frm_array = arr.array('B', [int(l * 256)]) * height * width * 3
            p.stdin.write(frm_array)
    else:
        output, errors = p.communicate()
        print(errors)


if __name__ == '__main__':

    description = """Creates a video file with an ENF signal. Parameters are currently hard-coded:\n
    - Length 120 seconds
    - Format 1080x1920
    - Grid frequency 50 Hz
    """
    parser = argparse.ArgumentParser(
        prog='fm',
        description=description
        )
    parser.add_argument('filename')
    args = parser.parse_args()

    t, sig_mod, delta_phi = gen_signals()
    lum = gen_lum_signal(sig_mod)
    gen_raw_video_file(lum, args.filename)
