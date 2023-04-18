#%% import modules
import pandas as pd
#from sep_util import read_file
import numpy as np
import matplotlib.pyplot as plt

# import utility functions
import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import remove_outliers, split_P_S_dataframe, filter_by_channel_number, filter_by_magnitude, filter_by_snr
from utility.regression import store_regression_results, fit_regression_iteration

def split_P_and_S_for_regression(peak_amplitude_df, min_channel, M_threshold, snr_threshold):
    """Split the P and S wave measurement for regression"""
    peak_amplitude_df_P, peak_amplitude_df_S = split_P_S_dataframe(peak_amplitude_df)
    peak_amplitude_df_P = filter_by_channel_number(peak_amplitude_df_P, min_channel)
    peak_amplitude_df_P = filter_by_magnitude(peak_amplitude_df_P, M_threshold)
    peak_amplitude_df_P = filter_by_snr(peak_amplitude_df_P, snr_threshold, 'snrP')

    peak_amplitude_df_S = filter_by_channel_number(peak_amplitude_df_S, min_channel)
    peak_amplitude_df_S = filter_by_magnitude(peak_amplitude_df_S, M_threshold)
    peak_amplitude_df_S = filter_by_snr(peak_amplitude_df_S, snr_threshold, 'snrS')
    return peak_amplitude_df_P, peak_amplitude_df_S


#%% 
# Apply to single array
results_output_dirs = ["/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate",
                       "/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South",
                       "/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North", 
                       "/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate",
                       "/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate",
                       "/kuafu/yinjx/Olancha/peak_ampliutde_scaling_results_strain_rate/New",
                       "/kuafu/yinjx/Olancha/peak_ampliutde_scaling_results_strain_rate/Old",
                       "/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/"]
                       #TODO: Sanriku needs more work!

M_threshold_list = [[2, 10], [2, 10], [2, 10], [2, 10], [2, 10], [0, 10], [0, 10], [2, 10]]
snr_threshold_list = [10, 10, 10, 10, 5, 5, 5, 20]
min_channel = 100

weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise

for i_region in [0]:#range(len(results_output_dirs)):
    M_threshold = M_threshold_list[i_region]
    snr_threshold = snr_threshold_list[i_region]
    results_output_dir = results_output_dirs[i_region]
    print(results_output_dir)

    regression_results_dir = results_output_dir + f'/iter_regression_results_smf{weight_text}_{min_channel}_channel_at_least_test'
    mkdir(regression_results_dir)

    peak_amplitude_df = pd.read_csv(results_output_dir + '/peak_amplitude_events/calibrated_peak_amplitude.csv')
    peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']

    if 'Sanriku' in results_output_dir: # some special processing for Sanriku data
        peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.QA == 'Yes']
        peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 4130].index)
        #peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 1580].index)



    peak_amplitude_df = peak_amplitude_df[['event_id', 'magnitude', 'depth_km', 'channel_id', 'distance_in_km', 
                                        'snrP', 'snrS', 'peak_P', 'peak_S', 'region']] 
                                    
    # to remove some extreme values
    peak_amplitude_df = remove_outliers(peak_amplitude_df, outlier_value=1e4)
    peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.peak_P<=0].index)
    peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.peak_S<=0].index)

    # Remove some bad data (clipped, poor-quality)
    if 'Ridgecrest' in results_output_dir:
        peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id.isin([38548295.0, 39462536.0])].index)
    if 'North' in results_output_dir:
        peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 73584926].index)
    if 'South' in results_output_dir:
        peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 73584926].index)
    if ('Olancha' in results_output_dir) and ('Old' in results_output_dir):
        peak_amplitude_df = remove_outliers(peak_amplitude_df, outlier_value=40)
    if 'Arcata' in results_output_dir:
        # good_events_list = [73736021, 73739276,  
        # 73743421, 73747016, 73751651,
        # 73757961, 73758756]#73741131,73747806, 73748011, 73755311, 73740886,73735891, 73753546, 73747621, 73739346,
        # good_events_list = [73736021, 73739276, 73747016, 73751651, 73757961, 73758756, 73739346, 73743421, 
        # 73735891, 73741131, 73747621, 73747806, 73748011, 73753546]
        # peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.event_id.isin(good_events_list)]


        event_id_fit_P = [73736021, 73747016, 73735891, 73747621, 73747806, 73748011, 73739346, 73743421, 73741131] #[73736021, 73747016, 73747621, 73743421] 
        event_id_fit_S = [73736021, 73747016, 73735891, 73747621, 73747806, 73748011, 73739346, 73743421, 73741131] 
        peak_amplitude_df_P = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit_P)]
        peak_amplitude_df_S = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit_S)]


    n_iter = 50
    rms_epsilon = 0.1 # percentage of rms improvement, if smaller, stop iteration


    # use P and S separately to do the regression
    if 'Arcata' not in results_output_dir:
        peak_amplitude_df_P, peak_amplitude_df_S = split_P_and_S_for_regression(peak_amplitude_df, min_channel, M_threshold, snr_threshold)
    
    try:
        regP, site_term_df_P, fitting_rms_P = fit_regression_iteration(peak_amplitude_df_P, wavetype='P', weighted=weighted, 
                                    n_iter=n_iter, rms_epsilon=rms_epsilon)
    except:
        print('P regression is unavailable, assign Nan and None')
        regP = None
        site_term_df_P = pd.DataFrame(columns=['region', 'channel_id', 'site_term_P'])

    try:
        regS, site_term_df_S, fitting_rms_S = fit_regression_iteration(peak_amplitude_df_S, wavetype='S', weighted=weighted, 
                                    n_iter=n_iter, rms_epsilon=rms_epsilon)
    except:
        print('S regression is unavailable, assign Nan and None')
        regS = None
        site_term_df_S = pd.DataFrame(columns=['region', 'channel_id', 'site_term_S'])

    # merge the site term
    site_term_df = pd.merge(site_term_df_P, site_term_df_S, how='outer', left_on=['channel_id', 'region'], right_on=['channel_id', 'region'])

    # store the regression results
    results_file_name = "regression_combined_site_terms_iter"
    store_regression_results(regP, regression_results_dir, results_filename=f"/P_{results_file_name}")
    store_regression_results(regS, regression_results_dir, results_filename=f"/S_{results_file_name}")
    site_term_df.to_csv(regression_results_dir + f'/site_terms_iter.csv', index=False)


# %%
