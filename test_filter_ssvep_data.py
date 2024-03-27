"""
    Definition:
    ----------
    Calls functions defined in filter_ssvep_data.py.
    Answers questions of the Lab 4
    
    Authors
    -----------
    Ashley Heath and Lute Lillo
"""

from import_ssvep_data import load_ssvep_data as load
from import_ssvep_data import plot_raw_data as plot
from filter_ssvep_data import make_bandpass_filter, filter_data, get_envelope, plot_ssvep_amplitudes, plot_filtered_spectra

subject = 2
data_directory = "SSVEP_data"

# Part 1
data = load(subject, data_directory)

# Part 2
filter_coeff_1 = make_bandpass_filter(13, 15, filter_order=1000, fs=1000)
filter_coeff_2 = make_bandpass_filter(12, 14, filter_order=1000, fs=1000)


# Part 3
filtered_data_1 = filter_data(data, filter_coeff_1)
filtered_data_2 = filter_data(data, filter_coeff_2)

# Part 4
envelope_1 = get_envelope(data, filtered_data_1, "3", None)
envelope_2 = get_envelope(data, filtered_data_2, "5", None)

# # Part 5
channel_to_plot = 29 # Oz channel
plot_ssvep_amplitudes(data, envelope_1, envelope_2, channel_to_plot, 12, 15, subject)

# Part 6
channels_to_plot= [4, 29] #index of channels where FZ and OZ is the name respectively
plot_filtered_spectra(data, filtered_data_1, envelope_1, channels_to_plot, subject)


"""
    Questions of Lab 4
    ---------
    
    Part 2
    ---------
    A) How much will 12Hz oscillations be attenuated by the 15Hz filter? How much will 15Hz
    oscillations be attenuated by the 12Hz filter?
    
    
    B) Experiment with higher and lower order filters. Describe how changing the order changes
    the frequency and impulse response of the filter.
    
    
    Part 5
    ---------
    Describe what you see. What do the two envelopes do when the stimulation frequency changes?
    
    How large and consistent are those changes?
    
    Are the brain signals responding to the events in the way you'd expect?
    
    Check some other electrodes - which electrodes respond in the same way and why?
    
    
    Part 6
    ----------
    1. Why does the overall shape of the spectrum change after filtering?
    
    2. In the filtered data on Oz, why do 15Hz trials appear to have less power than 12Hz trials at
    most frequencies?
    
    
    3. In the envelope on Oz, why do we no longer see any peaks at 15Hz?
    

"""
