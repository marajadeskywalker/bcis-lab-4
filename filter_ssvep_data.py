"""
    Definition:
    ----------
    Defines functions to construct a filter according to the input specifications,
    calculate the envelope from the filtered and unfiltered data,
    plot amplitudes and envelopes against event times,
    and plot the average power spectra across epochs for the data.
    
    Authors
    -----------
    Ashley Heath and Lute Lillo
"""

import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import bci_filtering_plot as bfp
from import_ssvep_data import get_frequency_spectrum, epoch_ssvep_data, epoch_filtered_data, plot_power_spectrum

def make_bandpass_filter(low_cutoff, high_cutoff, filter_type="hann", filter_order=10, fs=1000):
    """
        Definition:
        ----------
            Create a Filter and plot the impulse response and frequency response of your filter
        
        Parameters:
        ----------
            - low_cutoff (int): the lower cutoff frequency (in Hz).
            - high_cutoff (int): the higher cutoff frequency (in Hz)
            - filter_type (str): the filter type
            - filter_order (int): the filter order
            - fs (int): the sampling frequency in Hz
            
        Returns:
        ----------
            - filtered_data: data dictionary after having applied the filter forwards
            and backwards in time to each channel in the raw data.
    """
    
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
    
    plt.savefig(f"plots/Limit_{filter_type}_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}.png")
    
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
            Extract the envelope surrounding the waves of the amplitude of oscillations 
            usign the Hilbert Transform. This envelope tends to surf along the top of the
            wave, connecting all the peaks. It reflects the wave's amplitude
            
        Parameters:
        ----------
            - data (dict): the raw data dictionary.
            - filtered_data (array | Shape: (n_channels, eeg_data_points)): the filtered data.
            - channel_to_plot (str): which channel to plot.
            - ssvep_frequency (str): the SSVEP frequency being isolated.
            
        Returns:
        ----------
            - envelope (array | Shape (n_channels, n_time_points)): Amplitude of oscillations.
    
    """
    eeg = data['eeg']  
    fs = data['fs']
    channels = data['channels']
    
     # Create time variable
    number_of_samples = np.shape(eeg[0]) # number of samples in one session
    eeg_end_time = 1/fs * number_of_samples[0] # time point when the session ends in seconds
    time = np.linspace(start= 0, stop=eeg_end_time, num=number_of_samples[0]) 
    amplitude_envelope = []
    int_ch_plot = int(channel_to_plot)
    
    # Get the envelope for all channels
    for channel_idx, channel in enumerate(channels):

        filtered_signal_channel = filtered_data[channel_idx]
        analytical_signal = signal.hilbert(filtered_signal_channel)
        amplitude_envelope.append(np.abs(analytical_signal))

    # Plot for the specific channel
    if channel_to_plot is not None:
        figure, axis = plt.subplots(1,1, figsize=(15, 7)) 
        axis.plot(time, filtered_data[int_ch_plot], label="Filtered signal")
        axis.plot(time, amplitude_envelope[int_ch_plot], label="Envelope")
        axis.set_xlim(147, 164)
        
        # axis.set_ylim(-10, 10)
        axis.set_xlabel('time (s)')
        axis.set_ylabel('voltage (uV)')
        if ssvep_frequency is not None:
            axis.set_title(f"{ssvep_frequency}Hz BPF Data")
        else:
            axis.set_title("Unknown BPF Data")
        axis.grid(True)
       
        plt.savefig(f'plots/Channel_{channel_to_plot}-{ssvep_frequency}Hz.png')

    return amplitude_envelope

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot, ssvep_freq_a, ssvep_freq_b, subject):
    """
        Definition:
        ----------
            Plot the 12Hz and 15Hz envelopes alongside the task events to get a better sense of whether they are 
            responding to task events.
        
        Parameters:
        ----------
            - data (dict) the raw data dictionary,
            - envelope_a (array | Shape (n_channels, n_time_points)): Amplitude of oscillations at the first frequency
            - envelope_b (array | Shape (n_channels, n_time_points)): Amplitude of oscillations at the second frequency
            - channel_to_plot (str): which channel to plot.
            - ssvep_freq_a (int): the SSVEP frequency being isolated in the first envelope.
            - ssvep_freq_b (int): the SSVEP frequency being isolated in the second envelope.
            - subject (int): the subject number.
            
        Returns:
        ----------
            None
    """
    
    #define variables
    fs = data['fs'] # frequency in samples per second
    eeg_data = data['eeg'] # data in volts
    event_samples = data['event_samples'] # when each event occured
    event_durations = data['event_durations'] # how long each event occured for
    event_types = data['event_types'] # frequency of flickering
    
    # Create time variable
    number_of_samples = np.shape(eeg_data[0]) # number of samples in one session
    eeg_end_time = 1/fs * number_of_samples[0] #time point when the session ends in seconds
    time = np.linspace(start= 0, stop=eeg_end_time, num=number_of_samples[0]) 

    # Create Figure    
    fig, axs = plt.subplots(2, 1, sharex= True, figsize=(12,6))
    
    # Get start and end time.
    start_time = event_samples / fs
    end_time = (event_samples + event_durations) / fs

    # Plot event times and Hz. SUBPLOT 1
    for event_index, event in enumerate(event_samples):
        axs[0].plot(start_time[event_index], event_types[event_index], 'o', color='blue')
        axs[0].plot(end_time[event_index], event_types[event_index], 'o', color='blue')
        axs[0].plot([start_time[event_index], end_time[event_index]], [event_types[event_index], event_types[event_index]], color='blue')
    
    axs[0].set_xlabel('time (s)')
    axs[0].set_ylabel('Flash Frequency')
    
    int_ch_plot = int(channel_to_plot)
    # Plot envelopes
    axs[1].plot(time, envelope_a[int_ch_plot], label=f"{ssvep_freq_a}Hz", color='blue')
    axs[1].plot(time, envelope_b[int_ch_plot], label=f"{ssvep_freq_b}Hz", color='green')
    axs[1].legend()
    axs[1].set_xlim(200, 300)
    axs[1].set_ylabel('Voltage (uV)')
    
    plt.suptitle(f'SSVEP S{subject} Amplitudes - Channel {channel_to_plot}')
    plt.tight_layout()
    plt.savefig(f"plots/Channel_{channel_to_plot}_Subject_{subject}_{ssvep_freq_a}Hz-{ssvep_freq_b}Hz_envelope_comparison_limit.png")
    
# Part 6
def plot_filtered_spectra(data, filtered_data, envelope, channels_to_plot, subject):
    """
        Definition:
        ----------
            Plot the average power spectra across epochs on electrodes Fz and Oz at 3 stages of our analysis: Raw, Filtered and Envelope.
        
        Parameters:
        ----------
            - data (dict) the raw data dictionary,
            - envelope (array | Shape (n_channels, n_time_points)): Amplitude of oscillations at the first frequency
            - filtered_data (array | Shape: (n_channels, eeg_data_points)): the filtered data.
            - channels_to_plot (array): which channels to plot.
            - subject (int): the subject number.
            
        Returns:
        ----------
            None
    """
    
    # Use cases from Lab3 
    channels = data['channels']
    fs = data['fs']

    fig, axs = plt.subplots(3, 2, sharex= True, figsize=(12, 6))
    spectrum_db_12Hz = []
    spectrum_db_15Hz = []
    fft_frequencies = []
    
    # Plots raw data
    eeg_epochs, _, is_trial_15Hz = epoch_ssvep_data(data)
    egg_epochs_fft, fft_frequencies_raw = get_frequency_spectrum(eeg_epochs, fs) # eeg_epochs_fft -> (20, 32, 10001)
    raw_spectrum_db_12Hz, raw_spectrum_db_15Hz = plot_power_spectrum(egg_epochs_fft, fft_frequencies_raw, is_trial_15Hz, channels, channels_to_plot, subject)
    spectrum_db_12Hz.append(raw_spectrum_db_12Hz)
    spectrum_db_15Hz.append(raw_spectrum_db_15Hz)
    fft_frequencies.append(fft_frequencies_raw)
    
    # Plots filtered_data
    eeg_epochs, _, is_trial_15Hz = epoch_filtered_data(data, filtered_data, epoch_start_time=0, epoch_end_time=20)
    egg_epochs_fft, fft_frequencies_filt = get_frequency_spectrum(eeg_epochs, fs)
    filtered_spectrum_db_12Hz, filtered_spectrum_db_15Hz = plot_power_spectrum(egg_epochs_fft, fft_frequencies_filt, is_trial_15Hz, channels, channels_to_plot, subject)
    spectrum_db_12Hz.append(filtered_spectrum_db_12Hz)
    spectrum_db_15Hz.append(filtered_spectrum_db_15Hz)   
    fft_frequencies.append(fft_frequencies_filt)
    
    # Plots envelope
    envelope = np.array(envelope) # Adjust to make it work w/ legacy lab fns
    eeg_epochs, _, is_trial_15Hz = epoch_filtered_data(data, envelope, epoch_start_time=0, epoch_end_time=20)
    egg_epochs_fft, fft_frequencies_env = get_frequency_spectrum(eeg_epochs, fs)
    envelope_spectrum_db_12Hz, envelope_spectrum_db_15Hz = plot_power_spectrum(egg_epochs_fft, fft_frequencies_env, is_trial_15Hz, channels, channels_to_plot, subject)
    spectrum_db_12Hz.append(envelope_spectrum_db_12Hz)
    spectrum_db_15Hz.append(envelope_spectrum_db_15Hz)  
    fft_frequencies.append(fft_frequencies_env)
    
    spectrum_db_12Hz_t = np.array(spectrum_db_12Hz)
    fft_frequencies_t = np.array(fft_frequencies)
    
    # Create subplots
    ch_str = ['FZ', 'OZ']
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    stage_names = ['Raw', 'Filtered', 'Envelope']
    for channel_idx, channel in enumerate(channels_to_plot):
        for i in range(3):
            axs[channel_idx, i].plot(fft_frequencies[i], spectrum_db_12Hz[i][channel_idx], label='12Hz Trials')
            axs[channel_idx, i].plot(fft_frequencies[i], spectrum_db_15Hz[i][channel_idx], label='15Hz Trials')
            axs[channel_idx, i].set_title(f'Stage {stage_names[i]} - {ch_str[channel_idx]}')
            axs[channel_idx, i].set_xlabel('Power (dB)')
            axs[channel_idx, i].set_ylabel('Power (dB)')
            axs[channel_idx, i].axvline(x=12, linestyle='--', color='gray')
            axs[channel_idx, i].axvline(x=15, linestyle='--', color='gray')
            axs[channel_idx, i].legend()
  
    plt.tight_layout()
    # #save plot
    plt.savefig(f'plots/power_spectrum/S_{subject}_power_spectra_filtered.png')
