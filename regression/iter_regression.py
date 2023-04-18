#%%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../')

from utility.general import mkdir
from utility.processing import (
    combined_regions_for_regression,
    filter_event_first_order,
    remove_outliers,
    split_P_S_dataframe,
    filter_by_channel_number,
    filter_by_magnitude,
    filter_by_snr
)
from utility.regression import (
    store_regression_results,
    fit_regression_iteration
)

def split_P_and_S_for_regression(peak_amplitude_df, min_channel, M_threshold, snr_threshold):
    """
    Split the P and S wave measurement for regression
    """
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)
    peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
    peak_amplitude_df_P = filter_by_magnitude(peak_amplitude_df_P, M_threshold)
    peak_amplitude_df_P = filter_by_snr(peak_amplitude_df_P, snr_threshold, 'snrP')

    peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)
    peak_amplitude_df_S = filter_by_magnitude(peak_amplitude_df_S, M_threshold)
    peak_amplitude_df_S = filter_by_snr(peak_amplitude_df_S, snr_threshold, 'snrS')

    return peak_amplitude_df_P, peak_amplitude_df_S

#%% 
# some parameters
snr_threshold = 10
min_channel = 100 # do regression only on events recorded by at least 100 channels

# result directory
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
peak_file_name = 'peak_amplitude_multiple_arrays.csv'
mkdir(results_output_dir)

#%% 
# Preprocess the data file: combining different channels etc.
preprocess_needed = False  # If true, combined different regions data to produce the combined data file

peak_file_list = [
    '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv',
    '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events/calibrated_peak_amplitude.csv',
    '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events/calibrated_peak_amplitude.csv'
]

region_list = ['ridgecrest', 'mammothS', 'mammothN']
M_threshold_list = [[2, 10], [2, 10], [2, 10]]

M_threshold = [2, 10]

if preprocess_needed: 
    peak_amplitude_df = combined_regions_for_regression(peak_file_list)
    peak_amplitude_df=filter_event_first_order(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)

    # remove a clipped event 73584926 in the Mammoth data set
    peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 73584926].index)

    # remove a clipped event 38548295 and 39462536 in the Ridgecrest data set
    peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id.isin([38548295.0, 39462536.0])].index)

    peak_amplitude_df.to_csv(results_output_dir + f'/{peak_file_name}', index=False)


#%%
# Set up output directory and load data
weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise ValueError("Invalid weight type. Please use 'ols' or 'wls'")

regression_results_dir = results_output_dir + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least_test'
mkdir(regression_results_dir)

# Load and filter data
peak_amplitude_df = pd.read_csv(results_output_dir + f'/{peak_file_name}')
peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
peak_amplitude_df = peak_amplitude_df[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km', 'snrP', 'snrS', 'peak_P', 'peak_S', 'region']]

# Remove outliers and invalid values
peak_amplitude_df = remove_outliers(peak_amplitude_df, outlier_value=1e4)
peak_amplitude_df = peak_amplitude_df[peak_amplitude_df['peak_P'] > 0]
peak_amplitude_df = peak_amplitude_df[peak_amplitude_df['peak_S'] > 0]

#%% 
# Iteratively fitting
# Set up iteration parameters
n_iter = 20
rms_epsilon = 0.1 # percentage of rms improvement, if smaller, stop iteration

# Split P and S waves for regression
peak_amplitude_df_P, peak_amplitude_df_S = split_P_and_S_for_regression(peak_amplitude_df, min_channel, M_threshold, snr_threshold)

# Perform regression for P and S waves separately
try:
    regP, site_term_df_P, fitting_rms_P = fit_regression_iteration(peak_amplitude_df_P, wavetype='P', weighted=weighted,  
                                n_iter=n_iter, rms_epsilon=rms_epsilon)
except:
    print('P regression is unavailable, assign NaN and None')
    regP = None
    site_term_df_P = pd.DataFrame(columns=['region', 'channel_id', 'site_term_P'])

try:
    regS, site_term_df_S, fitting_rms_S = fit_regression_iteration(peak_amplitude_df_S, wavetype='S', weighted=weighted, 
                                n_iter=n_iter, rms_epsilon=rms_epsilon)
except:
    print('S regression is unavailable, assign NaN and None')
    regS = None
    site_term_df_S = pd.DataFrame(columns=['region', 'channel_id', 'site_term_S'])

# Merge site terms for P and S waves
site_term_df = pd.merge(site_term_df_P, site_term_df_S, how='outer', left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])

# Store regression results
results_file_name = "regression_combined_site_terms_iter"
store_regression_results(regP, regression_results_dir, results_filename=f"/P_{results_file_name}")
store_regression_results(regS, regression_results_dir, results_filename=f"/S_{results_file_name}")
site_term_df.to_csv(regression_results_dir + f'/site_terms_iter.csv', index=False)




# %%
