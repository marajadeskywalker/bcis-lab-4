# Import statements
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

#%% Part 1: Load the Data
def load_ssvep_data(subject, data_directory):
    """
    Description
    -----------
    This function imports the SSVEP data for a specific subject from a given directory as a dictionary.

    Parameters
    ----------
    subject : integer, the subject of the experiment whose data will be imported.
    data_directory : string, the relative filepath to the directory where the SSVEP data file for the specified subject is located. 

    Returns
    -------
    data_dict : dictionary, containing the SSVEP data of the intended subject.

    """
    data_dict = np.load(f'{data_directory}/SSVEP_S{subject}.npz', allow_pickle=True)

    return data_dict

#%% Part 2: Plot the Data
def plot_raw_data(data, subject, channels_to_plot):
    """
    Description
    -----------
    Plots the data of several particular channels/electrodes on the same graph, along with a separate plot of flash frequency.

    Parameters
    ----------
    data : dictionary, the dictionary object containing the data of the subject who is being plotted
    subject : integer, the subject of the experiment being plotted
    channels_to_plot : N x 1 array of integers, where N is the number of channels being plotted. This array contains the indices of all channels that
    will be plotted by the function.
    
    Returns
    -------
    None

    """
    #pull relevant values out of the dictionary
    eeg = data['eeg']       
    fs = data['fs']
    event_samples = data['event_samples']
    event_durations = data['event_durations']
    event_types = data['event_types']

    # Create figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex='all')
    plt.xlim(0, 100)

    # Plot event start and end times and types
    axs[0].set_title('Event Times')
    axs[0].set_ylabel('Flash Frequency')
    for i, (start, duration, event_type) in enumerate(zip(event_samples, event_durations, event_types)):
        axs[0].plot([start / fs, (start + duration) / fs], [event_type, event_type])

    # Plot raw data from specified channels
    axs[1].set_title('Raw EEG Data')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Voltage (uV)')
    for channel_name in channels_to_plot:
        channel_index = np.where(data['channels'] == channel_name)[0][0]
        axs[1].plot(np.arange(eeg.shape[1]) / fs, eeg[channel_index], label=channel_name)
    axs[1].legend()

    # Save the figure
    plt.savefig(f'SSVEP_S{subject}_rawdata')
    plt.show()

#%% Part 3: Extract the Epochs
def epoch_ssvep_data(data_dict, epoch_start_time=0, epoch_end_time=20):
    """
    Description
    -----------
    Divides the raw EEG data into epochs based on the events, with the given start and end times.

    Parameters
    ----------
    data_dict : dictionary, the dictionary object containing the SSVEP data that will be epoched
    epoch_start_time : int, the start time (in seconds) of the epoch relative to the initial event. Default: 0.
    epoch_end_time : int, the end time (in seconds) of the epoch relative to the initial event. Default: 0.
    
    Returns
    -------
    eeg_epochs : X x Y x Z array of floats, where X is the number of epochs, Y is the number of channels and Z is the number of 
    samples in each epoch. Contains the SSVEP data of the dictionary, reshaped and divided into epochs for each event.
    epoch_times : N x 1 array of floats, where N is the number of samples in each epoch. Contains the time, in seconds, that each
    sample the epoch occurred at relative to the original event.
    is_trial_15Hz : N x 1 array of booleans, where N is the number of epochs. Values are true if the epoch at that index flashed at 15Hz, 
    false otherwise.
    """
    #pull relevant values out of the dictionary
    event_samples = data_dict['event_samples']
    event_types = data_dict['event_types']
    eeg_data = data_dict['eeg']
    fs = data_dict['fs']
    
    #calculate other values based on this data
    channel_count = len(data_dict['channels'])
    epoch_count = len(event_samples)
    epoch_total_seconds = epoch_end_time - epoch_start_time
    samples_per_epoch = int(fs * (epoch_total_seconds))
    
    #calculate is_trial_15Hz via simple boolean comparison
    is_trial_15Hz = (event_types == '15hz')

    #use a numpy function to calculate even spacing for the appropriate number of samples in the time allotted
    epoch_times = np.zeros(samples_per_epoch)
    epoch_times = np.linspace(epoch_start_time, epoch_end_time, num=samples_per_epoch)
    
    #reshape the raw data into a three-dimensional array with epoch as the first dimension
    eeg_epochs = np.zeros((epoch_count, channel_count, samples_per_epoch))
    for epoch_index in range(epoch_count):
        for channel_index in range(channel_count):
            start_offset = (event_samples[epoch_index]+int(epoch_start_time*fs))
            for sample_index in range(samples_per_epoch):
                epoch_start_sample = sample_index + start_offset
                eeg_epochs[epoch_index, channel_index, sample_index] = eeg_data[channel_index, epoch_start_sample]
    return eeg_epochs, epoch_times, is_trial_15Hz

#%% Part 4: Take the Fourier Transform
def get_frequency_spectrum(eeg_epochs, fs):
    """
    Description
    -----------
    Conducts the fourier transformation on the epoched data for the given sampling frequency.

    Parameters
    ----------
    eeg_epochs : X x Y x Z array of floats, where X is the number of epochs, Y is the number of channels and Z is the number of 
    samples in each epoch. Contains the epoched SSVEP data, reshaped and divided into epochs for each event.
    fs : int, the sampling frequency in number of samples per second.
    
    Returns
    -------
    eeg_epochs_fft : X x Y x Z array of floats, where X is the number of epochs, Y is the number of channels and Z is half the number of samples in each epoch,
    rounded down, + 1. Contains the real valued results of the fourier transform on the data.
    fft_frequencies: N x 1 array of floats, where N is half the number of samples in each epoch, rounded down, + 1. Contains the frequency for each column in eeg_epochs_fft.
    """
    
    eeg_epochs_fft = np.fft.rfft(eeg_epochs, axis=-1)
    
    fft_frequencies = np.fft.rfftfreq(eeg_epochs.shape[-1], d=1/fs)

    return eeg_epochs_fft, fft_frequencies

# HELPER FUNCTION MODIFIED FROM LAB3
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

    return spectrum_db_12Hz, spectrum_db_15Hz