from import_ssvep_data import load_ssvep_data as load
from import_ssvep_data import plot_raw_data as plot
from import_ssvep_data import epoch_ssvep_data
import numpy as np
from import_ssvep_data import get_frequency_spectrum
from filter_ssvep_data import make_bandpass_filter, filter_data, get_envelope, plot_ssvep_amplitudes

subject = 1
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

# Part 5
channel_to_plot = 29 # Oz channel
plot_ssvep_amplitudes(data, envelope_1, envelope_2, channel_to_plot, 12, 15, 3)

