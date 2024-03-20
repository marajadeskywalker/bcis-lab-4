"""
filter_ssvep_data.py
Defines functions to construct a filter according to the input specifications, calculate the envelope from the filtered and unfiltered data,
plot amplitudes and envelopes against event times, and plot the average power spectra across epochs for the data.
Authored by Ashley Heath and Lute Lillo
"""

import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def make_bandpass_filter(low_cutoff, high_cutoff, filter_order, fs, filter_type="hann"):
    #calculate input values
    nyquist_frequency = fs/2
    low_cutoff_adjusted = low_cutoff/nyquist_frequency
    high_cutoff_adjusted = high_cutoff/nyquist_frequency

    #get filter coefficients
    filter_coefficients = scipy.signal.firwin(numtaps=filter_order+1, cutoff=[low_cutoff_adjusted, high_cutoff_adjusted], window=filter_type, fs=fs)
    print(np.shape(filter_coefficients))
    #calculate impulse response
    impulse = np.zeros_like(filter_coefficients)
    print(np.shape(filter_coefficients))
    impulse[len(filter_coefficients)//2] = 1
    impulse_filtered = np.fft.fft(impulse) * filter_coefficients
    impulse_response = np.fft.ifft(impulse_filtered).real 

    #calculate frequency response
    frequency_response = scipy.signal.freqz(filter_coefficients, fs=fs)

    lag_axis = np.arange(-len(filter_coefficients) // 2, len(filter_coefficients) // 2) / fs
    figure, axis = plt.subplots(2,1)                         # ... and plot the results.
    axis[0].plot(lag_axis, impulse_response, label="impulse response")
    plt.show()
    #axis.xlabel()
    
    #axis.plot(lag_axis, impulse, label="impulse")

