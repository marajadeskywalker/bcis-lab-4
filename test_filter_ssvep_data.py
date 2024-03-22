from import_ssvep_data import load_ssvep_data as load
from import_ssvep_data import plot_raw_data as plot
from import_ssvep_data import epoch_ssvep_data
import numpy as np
from import_ssvep_data import get_frequency_spectrum
from filter_ssvep_data import make_bandpass_filter

subject = 1
data_directory = "SSVEP_data"

# Part 1
data = load(subject, data_directory)


# Part 2

filter_coeff_1 = make_bandpass_filter(11, 13, 1000)
filter_coeff_2 = make_bandpass_filter(12, 14, 1000)