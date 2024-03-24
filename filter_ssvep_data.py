"""
filter_ssvep_data.py
Defines functions to construct a filter according to the input specifications, calculate the envelope from the filtered and unfiltered data,
plot amplitudes and envelopes against event times, and plot the average power spectra across epochs for the data.
Authored by Ashley Heath and Lute Lillo
"""

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import bci_filtering_plot as bfp

def make_bandpass_filter(low_cutoff, high_cutoff, filter_order=10, fs=1000, filter_type="hann"):
    
    low_cutoff_adjusted = low_cutoff
    high_cutoff_adjusted = high_cutoff
    cutoff_array = [low_cutoff_adjusted, high_cutoff_adjusted]
    
    # Get filter coefficients
    filter_coefficients = signal.firwin(numtaps=filter_order+1, cutoff=cutoff_array,
                                        window=filter_type, pass_zero='bandpass', fs=fs)
    
    # Calculate impulse response
    impulse = np.zeros_like(filter_coefficients) # Shape (1001,)
    impulse[len(filter_coefficients)// 2] = 1
    
    # Get impulse response. Method 1
    h_t = signal.lfilter(filter_coefficients, a=1, x=impulse)

    # Calculate frequency response
    frequency_response, H_f = signal.freqz(filter_coefficients, a=1, fs=fs)

    lag_axis = (np.arange(-len(filter_coefficients) // 2, len(filter_coefficients) // 2)) / fs # Shape (1001,)
    
    figure, axis = plt.subplots(2,1, figsize=(12,10))                     
    axis[0].plot(lag_axis, h_t, label="impulse response")
    axis[0].set_title("impulse response")
    axis[0].set_ylabel('gain')
    axis[0].set_xlabel('time(s)')
    
    axis[1].plot(frequency_response, bfp.convert_to_db(H_f), label="frequency response")
    axis[1].set_title("frequency response")
    axis[1].set_xlabel('frequency (Hz)')
    axis[1].set_xlim(0, 60)
    axis[1].set_ylabel('amplitude (dB)')
    
    # plt.savefig(f"plots/Limit_{filter_type}_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}.png")
    
    return filter_coefficients


def filter_data(data, b):
    """
        Definition:
        ----------
            Apply filter through the function scipy.signal.filtfilt() to the raw data.
        
        Parameters:
        ----------
            - data (dict): the raw data dictionary.
            - b (array | Shape (n_filter_order)): the filter coefficients produced in function make_bandpass_filter().
            
        Returns:
        ----------
            - filtered_data: data dictionary after having applied the filter forwards
            and backwards in time to each channel in the raw data.
    """
    eeg = data['eeg']       

    filter_data = signal.filtfilt(b, a=1, x=eeg)
    
    return filter_data

def get_envelope(data, filtered_data, channel_to_plot=None, ssvep_frequency=None):
    """
        Definition:
        ----------
            
        
        Parameters:
        ----------
            - data (dict): the raw data dictionary.
            - filtered_data (array | Shape: (n_channels, eeg_data_points)): the filtered data.
            - channel_to_plot (str): which channel to plot.
            - ssvep_frequency (str): the SSVEP frequency being isolated.
            
        Returns:
        ----------
            - envelope (array | Shape (n_channels, n_time_points)): Amplitude of oscillations
    
    """
    if channel_to_plot is None:
        # Create new plot
        pass
    
    
    
    if ssvep_frequency is None:
        plt.savefig(f"plots/unknown_freq.png")
    # If the ssvep_frequency input is None, your plot’s title should state that the frequency being 
    # isolated is unknown. The function should return the amplitude of oscillations (on every channel at 
    # every time point) in an array called envelope.

    envelope = []
    return envelope
