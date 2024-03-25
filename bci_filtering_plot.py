#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bci_filtering_plots.py

Functions to help with plotting filters in the BCIs course.

Created on Thu Mar 21 10:02:40 2024
@author: djangraw
"""

import numpy as np
from matplotlib import pyplot as plt

def convert_to_db(X):
    """
    Convert a signal from linear scale to decibels.

    Parameters:
        X (array of size frequency_count): Input signal in linear scale.

    Returns:
        X_dB (array of size frequency_count): Signal converted to decibels.
    """
    
    X_power = np.abs(X)**2
    X_dB = 10*np.log10(X_power/np.max(X_power))
    return X_dB

def plot_filtering(x_t,X_f,h_t,H_f,y_t,Y_f,t,f,f_filter,title='',is_plot_db = False):
    """
    Plot the raw signal, filter response, and filtered signal in both time and frequency domains.

    Parameters:
        x_t (array of size sample_count): Raw signal in the time domain.
        X_f (array of size frequency_count): Raw signal in the frequency domain.
        h_t (array of size sample_count): Filter impulse response in the time domain.
        H_f (array of size filter_frequency_count): Filter magnitude response in the frequency domain.
        y_t (array of size sample_count): Filtered signal in the time domain.
        Y_f (array of size frequency_count): Filtered signal in the frequency domain.
        t (array of size sample_count): Time axis values.
        f (array of size frequency_count): Frequency axis values corresponding to X_f and Y_f.
        f_filter (array of size filter_frequency_count): Frequency axis values corresponding to H_f.
        title (str, optional): Title for the plot. Defaults to an empty string.
        is_plot_db (bool, optional): Whether to plot in decibels. Defaults to False.
    """
    # plot raw data    
    plt.figure(2,clear=True)
    ax_t = plt.subplot(3,2,1)
    plt.plot(t,x_t)
    plt.xlabel('t (s)')
    plt.ylabel('x(t)')
    plt.title('raw signal in time domain')
    plt.grid()
    
    ax_f = plt.subplot(3,2,2)
    if is_plot_db:
        plt.plot(f,convert_to_db(X_f))
    else:
        plt.plot(f,np.abs(X_f))
    plt.xlabel('f (Hz)')
    plt.ylabel('|X(f)|')
    plt.title('raw signal in freq domain')
    plt.grid()
    
    # plot filter
    plt.subplot(3,2,3,sharex=ax_t)
    plt.plot(t,h_t)
    plt.xlabel('t (s)')
    plt.ylabel('h(t)')
    plt.title('filter impulse response')
    plt.grid()
    
    plt.subplot(3,2,4,sharex=ax_f)
    if is_plot_db:
        plt.plot(f_filter,convert_to_db(H_f))
    else:
        plt.plot(f_filter,np.abs(H_f))
    plt.xlabel('f (Hz)')
    plt.ylabel('|H(f)|')
    plt.title('filter magnitude response')
    plt.grid()
    
    # plot filtered signal
    plt.subplot(3,2,5,sharex=ax_t)
    plt.plot(t,x_t,alpha=0.5,label='raw')
    plt.plot(t,y_t,label='filtered')
    plt.xlabel('t (s)')
    plt.ylabel('y(t)')
    plt.title('filtered data in the time domain')
    plt.legend()
    plt.grid()
    
    plt.subplot(3,2,6,sharex=ax_f)
    if is_plot_db:
        plt.plot(f,convert_to_db(X_f),alpha=0.5,label='raw')
        plt.plot(f,convert_to_db(Y_f),label='filtered')
    else:
        plt.plot(f,np.abs(X_f),alpha=0.5,label='raw')
        plt.plot(f,np.abs(Y_f),label='filtered')
    plt.xlabel('f (Hz)')
    plt.ylabel('|Y(f)|')
    plt.title('filtered data in the freq domain')
    plt.legend()
    plt.grid()
    
    plt.suptitle(title)
    plt.tight_layout()
    # plt.savefig("test.png")