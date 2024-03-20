from import_ssvep_data import load_ssvep_data as load
from import_ssvep_data import plot_raw_data as plot
from import_ssvep_data import epoch_ssvep_data
import numpy as np
from import_ssvep_data import get_frequency_spectrum

subject = 1
data_directory = '/Users/Max/Documents/BCIs/project-3/Lab-3/SsvepData'

# Part 1
data = load(subject, data_directory)
print(data['time'])