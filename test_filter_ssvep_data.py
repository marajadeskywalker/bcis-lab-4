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
print(data["channels"][13])
# Part 2
filter_coeff_1 = make_bandpass_filter(14, 16, filter_order=1000, fs=1000)
filter_coeff_2 = make_bandpass_filter(11, 13, filter_order=1000, fs=1000)


# Part 3
filtered_data_1 = filter_data(data, filter_coeff_1)
filtered_data_2 = filter_data(data, filter_coeff_2)

# Part 4
envelope_1 = get_envelope(data, filtered_data_1, "3", None)
envelope_2 = get_envelope(data, filtered_data_2, "5", None)

# Part 5
channel_to_plot = 29 # Oz channel 29
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
    
    The filters should both suppress most of the oscillations of the other frequency, because the passband is small enough for each filter to have
    minimal overlap and most of the other signal will be filtered out. However, 12Hz and 15Hz are close enough that because we are not using a naive
    rectangular filter with a hard cutoff like we discussed in class, some of the tail of each signal might slip in. 
    
    B) Experiment with higher and lower order filters. Describe how changing the order changes
    the frequency and impulse response of the filter.
    
    The more the order increases, the more the impulse response of the filter oscillates over the time domain. At very low order,
    it is almost a flat line, but at higher order it becomes more and more sinusoidal. By contrast, as order increases, frequency
    response moves more and more toward a single large peak in amplitude at 10Hz, while the rest of the function flattens out.
    
    
    Part 5
    ---------
    Describe what you see. What do the two envelopes do when the stimulation frequency changes?
    
    When the stimulation frequency changes, the voltage of the 15Hz envelope appears to fall noticeably, while the voltage of the 12Hz envelope
    remains steady.
    
    How large and consistent are those changes?
    
    This change is relatively large, constituting a voltage drop of about 0.4 to 0.2. It also relatively consistent. There are occasional higher
    spikes, but significantly fewer of them than before the frequency change.
    
    Are the brain signals responding to the events in the way you'd expect?
    I would have expected the 15Hz signal to increase in voltage rather than decrease once frequency of the signal increases. However,
    it's wholly possible that this intuition is wrong.
    
    
    Check some other electrodes - which electrodes respond in the same way and why?
    Electrodes that are close to the 29 / Oz electrode, like 28 / O1 or 30 / O2, display the same property. This is likely because 
    the SSVEP phenomena that we are examining in this lab appear mostly at the back of the brain in the primary visual cortex, so 
    electrodes in other places would not display any noticeable features.
    
    Part 6
    ----------
    1. Why does the overall shape of the spectrum change after filtering?
    
    The overall shape of the spectrum changes after filtering because most of the information not in the passband has been removed,
    and the power spectrum represents how much each specific frequency capures the structure of the data. If the filter takes out the data
    not fitting that specific frequency, then it won't appear in the spectrum, because it's no longer necessary to explain the data.
    
    2. In the filtered data on Oz, why do 15Hz trials appear to have less power than 12Hz trials at
    most frequencies?
    
    The 15Hz trials have less power than the 12Hz trials because more of the data's structure can be explained by, or depends on, the 12Hz trials.
    This is likely due to how the 12Hz frequency remains relatively constant across the time domain, but the 15Hz frequency changes noticeably
    after the frequency shift.
    
    3. In the envelope on Oz, why do we no longer see any peaks at 15Hz?
    
    We no longer see peaks in the envelope because the envelope only considers the smoothed-out peak values of the waveform rather than all of them. Consequently,
    the values smaller than the highest points on the waveform no longer appear on the power spectrum because they aren't important for defining the envelope.

"""
