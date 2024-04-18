#
# Generates a video file with an ENF pattern. The file is always 120 seconds
# long. Format is always 1080x1920.
#

import numpy as np
import subprocess
import array as arr
import argparse

#
# Parameters
#

duration = 120              # Duration in seconds
height = 1080
width = 1920


def gen_signals(fs: int, fc: int, duration: int):
    """Generates an ENF signal at the nominal grid frequency that is slightly
    frequency modulated.

    :param fc: Nominal grid frequency
    :param fs: Sampling frequency; must be an integer multiple of the video frame rate
    :param duration: Duration of the generated video
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
    sig_mod = np.abs(np.sin(phi))      # modulated signal

    return t, sig_mod


def gen_lum_signal(sig_mod, fs, fps):
    """Generate a sequence of luminance values from sig_mod.

    :param sig_mod: Series of samples of brightness values.
    :param fs: Sampling frequency
    :param fps: Video frame rate
    :returns lum: Array with one luminance value per frame.

    Downsample
    """
    samples_per_frame = fs // fps
    total_samples = len(sig_mod)
    lum = [np.average(sig_mod[t:t+samples_per_frame]) for t in range(0, total_samples, samples_per_frame)]

    assert np.shape(lum) == (total_samples // samples_per_frame, )
    return lum


def gen_raw_video_file(lum, filename, contrast=256, speed='medium'):
    """Pipe a raw video file to STDOUT, using the luminance values in lum.

    :param lum: Array with luminance values.
    :param filenam: Name of the file that ffmpeg generates.
    :param contrast: Contrast of the generated video file.

    The pixel format that is fed into ffmpeg is RGB 8:8:8.
    """

    assert contrast >= 0 and contrast < 256

    cmd = ["/usr/bin/ffmpeg", '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '1920x1080', '-r',
           '30', '-pix_fmt', 'rgb24', '-i', 'pipe:', '-c:v', 'libx264', '-preset', speed,
            '-qp', '0', '-y', filename]

    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if p.returncode is None:
        try:
            for l in lum:
                frm_array = arr.array('B', [int(l * contrast)]) * height * width * 3
                p.stdin.write(frm_array)
        except BrokenPipeError:
            print("Video generation aborted.")
    else:
        _, errors = p.communicate()
        print(errors)


if __name__ == '__main__':

    description = """Creates a video file with an ENF signal. Some parameters are hard-coded:
    Length 120 seconds, format is 1080x1920.
    """
    parser = argparse.ArgumentParser(
        prog='fm',
        description=description
        )
    parser.add_argument('filename', help="Name of the generated video file")
    parser.add_argument('--fps', default=30, help="Frames per second; default is 30 fps", type=int)
    parser.add_argument('--format', default="1080x1920", help="Format of the generated video file",
                        choices=["1080x1920"])
    parser.add_argument('--grid', default=50, help="Nominal grid frequency (Hz); default is 50 Hz",
                        choices=[50, 60])
    parser.add_argument('--contrast', default=128,
                        help="Contrast of the generated video {0..255}; default is 128",
                        type=int)
    args = parser.parse_args()

    fs = args.fps * 20

    t, sig_mod = gen_signals(fs, args.grid, duration)
    lum = gen_lum_signal(sig_mod, fs, args.fps)
    gen_raw_video_file(lum, args.filename, contrast=args.contrast)
