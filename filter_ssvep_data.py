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
from import_ssvep_data import get_frequency_spectrum, epoch_ssvep_data

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
    
    for channel_idx, channel in enumerate(channels):

        filtered_signal_channel = filtered_data[channel_idx]
        analytical_signal = signal.hilbert(filtered_signal_channel)
        amplitude_envelope.append(np.abs(analytical_signal))

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
       
        # plt.savefig(f'plots/Channel_{channel_to_plot}-{ssvep_frequency}Hz.png')

    return amplitude_envelope

def plot_ssvep_amplitudes(data, envelope_a, envelope_b, channel_to_plot, ssvep_freq_a, ssvep_freq_b, subject):
    """
        Definition:
        ----------
        
        
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
    
    # Plot envelopes
    axs[1].plot(time, envelope_a, label=f"{ssvep_freq_a}Hz", color='blue')
    axs[1].plot(time, envelope_b, label=f"{ssvep_freq_b}Hz", color='green')
    axs[1].legend()
    axs[1].set_xlim(200, 300)
    axs[1].set_ylabel('Voltage (uV)')
    
    plt.suptitle(f'SSVEP S{subject} Amplitudes - Channel {channel_to_plot}')
    plt.tight_layout()
    # plt.savefig(f"plots/Channel_{channel_to_plot}_Subject_{subject}_{ssvep_freq_a}Hz-{ssvep_freq_b}Hz_envelope_comparison_limit.png")
    
# Part 6
def plot_filtered_spectra(data, filtered_data, envelope, channels_to_plot, subject):
    """
        Definition:
        ----------
        
        
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

    # Plots raw data
    eeg_epochs, epoch_times, is_trial_15Hz = epoch_ssvep_data(data)
    egg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs) # eeg_epochs_fft -> (20, 32, 10001)
    raw_spectrum_db_12Hz, raw_spectrum_db_15Hz = plot_power_spectrum(egg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject)
    
    # Plots filtered_data # (32, 467580)
    eeg_epochs, epoch_times, is_trial_15Hz= epoch_filtered_data(data, filtered_data, epoch_start_time=0, epoch_end_time=20)
    egg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
    filtered_spectrum_db_12Hz, filtered_spectrum_db_15Hz = plot_power_spectrum(egg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject)

    # Plots envelope
    eeg_epochs, epoch_times, is_trial_15Hz = epoch_ssvep_data(envelope)
    egg_epochs_fft, fft_frequencies = get_frequency_spectrum(eeg_epochs, fs)
    envelope_spectrum_db_12Hz, envelope_spectrum_db_15Hz = plot_power_spectrum(egg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject)
    
    
    # Create Figure    
    fig, axs = plt.subplots(3, 2, sharex= True, figsize=(12, 6))
    for channel, channel_index in enumerate(channels_to_plot):
        print(channel)
        exit()
        axs[channel].plot(fft_frequencies, spectrum_db_12Hz[channel], label='12Hz Trials')
        axs[channel].plot(fft_frequencies, spectrum_db_15Hz[channel], label='15Hz Trials')
        axs[channel].axvline(x=12, linestyle='--', color='gray')
        axs[channel].axvline(x=15, linestyle='--', color='gray')
        #format plot
        axs[channel].set_title(f'Channel {channels[channel_index]} Power Spectrum (Subject {subject})')
        axs[channel].set_ylabel('Power (dB)')
        axs[channel].legend()
    
    
# Plotting power spectrum from Lab 3
#%% Part 5
def plot_power_spectrum(eeg_epochs_fft, fft_frequencies, is_trial_15Hz, channels, channels_to_plot, subject):
    """
        Calculate the mean power spectra for the specified channels
        and then, plot each in their own subplot.
        
        Parameters:
        - eeg_epochs_fft:  3D complex128 array,
            E x C x F where E is the epoch when the flashing occured, C is the 
            channels where the data was obtained, F is the frequency in hz
        - fft_frequencies: 1D complex 128 array
            Lx1 where L is the frequency corresponding to each column in the FFT
        - is_trial_15Hz: list
                an list in which is_trial_15Hz[i] is True if the light was flashing at 15Hz during a given epoch
        - channels:  1D Array of string, Cx1 where C is the name of each channel by electrode
        eeg: array of float, C x S where C is the number of channels and 
        S is the number of samples in a session
        - channels_to_plot: list
            list of channel indices to plot, default is ''
        - subject: Int
            Subject number that is of interest default is ''
        
        Returns:
        - spectrum_db_12Hz: list of elements per number of channels_to_plot
            each channel holds an 1D array of float: Px1 where P is the mean power spectrum of 12Hz trials in decibels
        - spectrum_db_15Hz: list of elements per number of channels_to_plot
            each channel holds an 1D array of float: Px1 where P is the mean power spectrum of 15Hz trials in decibels
    
    """
   
    # Initialize variables to store the mean power spectra
    spectrum_db_12Hz = []
    spectrum_db_15Hz = []
    
    # Create subplots
    fig, axs = plt.subplots(len(channels_to_plot), 1, figsize=(12, 8), sharex=True)
    
    # Iterate over each channel to plot
    for channel, channel_index in enumerate(channels_to_plot):

        # Initialize variables to store the power spectra for 12Hz and 15Hz trials
        power_12Hz = []
        power_15Hz = []

        # Iterate over each trial
        for frequency_index in range(eeg_epochs_fft.shape[0]):
            # Calculate absolute value of the spectrum
            spectrum_abs = np.abs(eeg_epochs_fft[frequency_index, channel_index, :])

            # Calculate power by squaring the spectrum
            power = spectrum_abs ** 2

            # Append power spectra for 12Hz and 15Hz trials
            if is_trial_15Hz[frequency_index]:
                power_15Hz.append(power)
            else:
                power_12Hz.append(power)

        # Take the mean across trials
        mean_power_12Hz = np.mean(power_12Hz, axis=0)
        mean_power_15Hz = np.mean(power_15Hz, axis=0)

        # Normalize the power spectra
        max_power = np.max([mean_power_12Hz, mean_power_15Hz])
        mean_power_12Hz_norm = mean_power_12Hz / max_power
        mean_power_15Hz_norm = mean_power_15Hz / max_power

        # Convert to decibel units
        spectrum_db_12Hz.append(10 * np.log10(mean_power_12Hz_norm))
        spectrum_db_15Hz.append(10 * np.log10(mean_power_15Hz_norm))       

       # Plot the mean power spectra
    #     axs[channel].plot(fft_frequencies, spectrum_db_12Hz[channel], label='12Hz Trials')
    #     axs[channel].plot(fft_frequencies, spectrum_db_15Hz[channel], label='15Hz Trials')
    #     axs[channel].axvline(x=12, linestyle='--', color='gray')
    #     axs[channel].axvline(x=15, linestyle='--', color='gray')
    #     #format plot
    #     axs[channel].set_title(f'Channel {channels[channel_index]} Power Spectrum (Subject {subject})')
    #     axs[channel].set_ylabel('Power (dB)')
    #     axs[channel].legend()

    # plt.xlabel('Frequency (Hz)')
    # plt.tight_layout()
    # #save plot
    # plt.savefig(f'plots/power_spectrum/SSVEP_S{subject}_Power Spectrum_Full_diffEpochStartEnd.png')

    return spectrum_db_12Hz, spectrum_db_15Hz

def epoch_filtered_data(data, eeg_data, epoch_start_time=0, epoch_end_time=20):
    fs = data['fs'] # 1000
    event_samples = data['event_samples'] # (20,)
    event_durations = data['event_durations'] # (20,) -> how long each event occured for
    event_types = data['event_types'] # (20,) -> frequency of flickering (12Hz or 15Hz)
    channels = data['channels'] # names of channels (32,)
    # Get the samples per epoch time dimension    
    samples_per_second = fs # Hz
    seconds_per_epoch = epoch_end_time - epoch_start_time
    samples_per_epoch = int(samples_per_second * seconds_per_epoch) 
    
    # Create empty 3D array of epochs. 
    eeg_epochs = np.zeros((len(event_samples), len(channels), samples_per_epoch)) # (20, 32, 20000)
    
    # Loop through samples to extract epochs    
    for event_index, event in enumerate(event_samples):
        before_flash = epoch_start_time * samples_per_epoch
        start_sample = int(event + before_flash) # When the event starts
        end_sample = int((event + event_durations[event_index]))
        eeg_epochs[event_index] = eeg_data[:, start_sample:end_sample]

    # Get the epoch times 1D of times relative to events
    epoch_times = np.arange(epoch_start_time, epoch_end_time, 1/samples_per_second ) # Shape (20000,)

    # Change 15Hz for True values
    is_trial_15Hz = [False if x == "12hz" else True for x in event_types]    
    
    return eeg_epochs, epoch_times, is_trial_15Hz