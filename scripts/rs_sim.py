#
# Simulates rolling shutter video processing
#

import numpy as np
import matplotlib.pyplot as plt


# The vertical resolution of the image sensor (number of scanlines) is vres. A
# real CMOS sensor scans rows sequentially for each image frame, with a pause
# of unknown length after scanning. This script simulates the output of the
# sensor, visualising the signal and its sprectrum.
#
# Simulation parameters are set here:

fgrid = 50                      # Grid frequency
fps = 35                        # Frames per second
vres = 640                      # Vertical resolution, number of scan lines
vidle = 100                      # Number idle scan lines per frame


def light_source_signal(t):
    """Get light source signal at time t, depending on the grid frequency fgrid.

    :param t: The time in seconds
    :returns: Luminosity (in arbitry units)
    """
    s = np.abs(np.sin(2 * np.pi * fgrid * t))
    return s


def sensor_output_signal(nframes):
    # t ist ein Array der Abtastzeitpunkte, wenn sie gleichmäßig über die
    # Bildzeit verteilt wären. Pro Bild sind das vres Werte, für n Bilder also
    # n * vres Werte, die Zeitdauer ist dann n/fps.
    #
    # Die Zeit für eine Zeile ist 1 / (fps * vres).
    t = np.arange(0, nframes/fps, 1/(fps * vres))

    # Array initialisieren
    s = np.zeros(nframes * vres)
    for frame in range(nframes):
        for row in range(vres):
            s[frame * vres + row] = light_source_signal(frame/fps +
                                                        1/fps * row/(vres+vidle))
    return t, s - np.mean(s)

fig = plt.figure(figsize=(7, 7), layout='constrained')
axs = fig.subplot_mosaic([["signal", "signal"],
                          ["magnitude", "log_magnitude"],
                          ['spectrum', 'spectrum']])

t, s = sensor_output_signal(100)

fft_ampl = np.abs(np.fft.fft(s))
fft_freq = np.fft.fftfreq(fft_ampl.shape[0], 1 / (fps * vres))
imax = np.argmax(fft_ampl)
print(f"Spectrum peak for {vidle} idle rows at {fft_freq[imax]} Hz")

# Plot the signal
axs["signal"].set_title("Signal")
axs["signal"].plot(t, s, color='C0')
axs["signal"].set_xlabel("Time (s)")
axs["signal"].set_ylabel("Amplitude")

# plot different spectrum types:
axs["magnitude"].set_title("Magnitude Spectrum")
axs["magnitude"].magnitude_spectrum(s, Fs=fps * vres, color='C1')

axs["log_magnitude"].set_title("Log. Magnitude Spectrum")
axs["log_magnitude"].magnitude_spectrum(s, Fs=fps * vres, scale='dB', color='C1')

axs['spectrum'].set_title('Spectrum')
axs['spectrum'].plot(fft_freq[0:1000], fft_ampl[0:1000])

plt.show()
