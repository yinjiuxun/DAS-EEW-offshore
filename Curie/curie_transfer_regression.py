#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import shutil
import statsmodels.api as sm
import random
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import filter_event_first_order
from utility.regression import fit_regression_transfer

#%% 
# define some local function for convenience
def split_fit_and_predict(N_event_fit, peak_amplitude_df):
    """Randomly choose a few events to fit"""
    event_id_all =  peak_amplitude_df.event_id.unique()
    random.shuffle(event_id_all)
    event_id_fit = event_id_all[:N_event_fit] # event id list for regression fit
    event_id_predict = event_id_all[N_event_fit:] # event id list for regression prediction

    peak_amplitude_df_fit = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit)]
    peak_amplitude_df_predict = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_predict)]

    return list(event_id_fit), peak_amplitude_df_fit, list(event_id_predict), peak_amplitude_df_predict

def specify_fit_and_predict(event_id_fit, event_id_predict, peak_amplitude_df):
    """Specify the fit and predict events """
    peak_amplitude_df_fit = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit)]
    peak_amplitude_df_predict = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_predict)]

    return peak_amplitude_df_fit, peak_amplitude_df_predict    

def transfer_fitting(regP_pre, regS_pre, peak_amplitude_df_fit, weighted):
    """Transfer scaling to obtain site terms"""
    site_term_P = fit_regression_transfer(peak_amplitude_df_fit, regP_pre, wavetype='P', weighted=weighted, M_threshold=M_threshold, snr_threshold=snr_threshold_transfer, min_channel=min_channel)
    site_term_S = fit_regression_transfer(peak_amplitude_df_fit, regS_pre, wavetype='S', weighted=weighted, M_threshold=M_threshold, snr_threshold=snr_threshold_transfer, min_channel=min_channel)

    # combine P and S
    site_term_df = pd.merge(site_term_P, site_term_S, on='channel_id', how='outer')
    site_term_df['region'] = peak_amplitude_df_fit.region.unique()[0]
    site_term_df = site_term_df.iloc[:, [0, 3, 1, 2]]
    return site_term_df

#%%
# some parameters
min_channel = 100 # do regression only on events recorded by at least 100 channels
weighted = 'ols' # Ordinary Linear Square regression

#%%
# Coefficients from previous results
previous_regression_dir = f'../data_files/transferred_regression'
regP_pre_path = previous_regression_dir + f"/P_regression_combined_site_terms_iter.pickle"
regS_pre_path = previous_regression_dir + f"/S_regression_combined_site_terms_iter.pickle"

regP_pre = sm.load(regP_pre_path)
regS_pre = sm.load(regS_pre_path)

# Curie
results_output_dir = '../results'
mkdir(results_output_dir)
snr_threshold_transfer = 10
M_threshold = [2, 10]
speficy_events = True
event_id_fit_P0 = [9007]
event_id_fit_S0 = [9007]
event_id_predict0 = [9001, 9006]

peak_amplitude_df = pd.read_csv('../data_files/peak_amplitude/calibrated_peak_amplitude.csv')
peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
peak_amplitude_df = filter_event_first_order(peak_amplitude_df, snr_threshold=snr_threshold_transfer, min_channel=min_channel, M_threshold=M_threshold)
event_id_all =  peak_amplitude_df.event_id.unique()

# if speficy_events:
peak_amplitude_df_fit, peak_amplitude_df_predict = specify_fit_and_predict(event_id_fit_P0, event_id_predict0, peak_amplitude_df)
# Transfer scaling to obtain site terms
site_term_df_P = transfer_fitting(regP_pre, regS_pre, peak_amplitude_df_fit, weighted)
site_term_df_P = site_term_df_P.drop(columns=['site_term_S'])

peak_amplitude_df_fit, peak_amplitude_df_predict = specify_fit_and_predict(event_id_fit_S0, event_id_predict0, peak_amplitude_df)
# Transfer scaling to obtain site terms
site_term_df_S = transfer_fitting(regP_pre, regS_pre, peak_amplitude_df_fit, weighted)
site_term_df_S = site_term_df_S.drop(columns=['site_term_P'])

site_term_df = pd.merge(left=site_term_df_P, right=site_term_df_S, how='outer', on=['channel_id', 'region'])

# make output directory and output results
results_output_dir = results_output_dir
regression_results_dir = results_output_dir + f'/transfer_regression_with_9007'
mkdir(regression_results_dir)

site_term_df.to_csv(regression_results_dir + '/site_terms_transfer.csv', index=False)

# output the event id list of fit and predict events
np.savez(regression_results_dir + '/transfer_event_list.npz', 
    event_id_fit_P=event_id_fit_P0, event_id_fit_S=event_id_fit_S0, event_id_predict=event_id_predict0)

# also copy the regression results to the results directory
shutil.copyfile(regP_pre_path, regression_results_dir + '/P_regression_combined_site_terms_transfer.pickle')
shutil.copyfile(regS_pre_path, regression_results_dir + '/S_regression_combined_site_terms_transfer.pickle')

# %%
