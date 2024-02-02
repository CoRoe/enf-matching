from scipy import signal, io
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

test_input = "001.wav"
grid_input = "/home/cro/Downloads/fnew-2023-12.csv"

# Load downsampled test signal
fs, test_signal = io.wavfile.read(test_input) 
#test_signal=test_signal[:,0]

recording_duration = int(len(test_signal)/fs)

# Load electricity grid reference signal
grid_table = pd.read_csv(grid_input)
grid_table['dtm'] = pd.to_datetime(grid_table['dtm'])
grid_times = grid_table['dtm'].to_numpy()
grid_freqs = grid_table['f'].to_numpy()

grid_temporal_resolution = (grid_times[1]-grid_times[0])/np.timedelta64(1000000000, 'ns') # A numpy trick to convert the interval to seconds

# Construct Bandpass Filter (Butterworth)

# Specify the expected grid frequency
grid_frequency = 50

# Filter parameters
filter_center_frequency = 2*grid_frequency # Often interested in the second harmonic i.e. 2*grid_frequency
half_bandwidth = 1 # half the width of the filter passband (in Hertz)
order = 4 # butterworth filter order

critical_frequencies = [filter_center_frequency - half_bandwidth, filter_center_frequency + half_bandwidth] 
bandpass_filter = signal.butter(order, critical_frequencies, btype='bandpass', output='sos', fs=fs)

# Set parameters for the STFT

nperseg = 10*fs # number of signal samples to include in each FFT (increases the STFT resolution in frequency and decreases it in time)
nfft = 1024*16 # Number of bins in the fft (this zero pads the signal and is a substitute for quadratic interpolation)

noverlap=nperseg-grid_temporal_resolution*fs # This ensures the temporal resolution of the STFT matches the reference data

# Run the signal processing pipeline
# Extract the peak frequency from the test signal (within the confines of the bandpass filter)

filtered_signal = signal.sosfilt(bandpass_filter, test_signal) # Apply the bandpass filter
print(f"fs={fs}, nfft={nfft}, noverlap={noverlap}, nperseg={nperseg}")
freqs,times,stft = signal.stft(filtered_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft) # Apply STFT
peak_freqs = [freqs[idx] for t in range(len(times)) if (idx := np.argmax(stft[:,t]))] # Extract peak for each point in time
signal_freqs = signal.medfilt(peak_freqs, kernel_size=29) # Apply a median filter, a larger kernel size means more smoothing

# Plot the STFT over the frequency range of interest
freqs_to_show = np.all([freqs>critical_frequencies[0]-1,freqs<critical_frequencies[1]+1], axis=0)

plt.figure(figsize=(10,5))
plt.pcolormesh(times, freqs[freqs_to_show], np.abs(stft[freqs_to_show,:]),vmin=0, vmax=0.2e6, shading='gouraud')
plt.plot(signal_freqs,color="white",alpha=0.75)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
