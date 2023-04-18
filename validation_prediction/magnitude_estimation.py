#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import filter_event_first_order
from utility.regression import predict_magnitude, get_mean_magnitude
from utility.plotting import plot_magnitude_seaborn

# set plotting parameters 
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 100,  # to adjust notebook inline plot size
    'axes.labelsize': 18, # fontsize for x and y labels (was 10)
    'axes.titlesize': 18,
    'font.size': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex':False,
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 
    'pdf.fonttype': 42 # Turn off text conversion to outlines
}

matplotlib.rcParams.update(params)

#%%
# some parameters
min_channel = 100 # do regression only on events recorded by at least 100 channels
M_threshold = [2, 10]
weighted = 'wls' # 'ols' or 'wls'
if weighted == 'ols':
    weight_text = '' 
elif weighted == 'wls':
    weight_text = '_weighted' 
else:
    raise

random_test = False # whether to run the random test for transfered scaling

results_output_dir_list = []
regression_results_dir_list = []
peak_file_name_list = []
result_label_list = []
M_threshold_list = []
snr_threshold_list = []
vmax_list = []
region_text_list = []

#%% # Set result directory
if not random_test:
    # ================== multiple arrays ================== 
    results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
    peak_file_name = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/peak_amplitude_multiple_arrays.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [120, 180] # for P and S
    region_text = 'California arrays'
    plot_type = 'histplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    # single arrays
    #  ================== Ridgecrest ================== 
    results_output_dir = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Ridgecrest/Ridgecrest_scaling/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [70, 100] # for P and S
    region_text = 'Ridgecrest'
    plot_type = 'histplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== Long Valley N ================== 
    results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North'
    peak_file_name = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/North/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [35, 50] # for P and S
    region_text = 'Long Valley North'
    plot_type = 'histplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== Long Valley S ================== 
    results_output_dir = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South'
    peak_file_name = '/kuafu/yinjx/Mammoth/peak_ampliutde_scaling_results_strain_rate/South/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [20, 30] # for P and S
    region_text = 'Long Valley South'
    plot_type = 'histplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== Sanriku fittd ================== 
    results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 5
    M_threshold = [2, 10]
    vmax = [2, 2] # for P and S
    region_text = 'Sanriku'
    plot_type = 'scatterplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== LAX fittd ================== 
    results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 10
    M_threshold = [2, 10]
    vmax = [2, 2] # for P and S
    region_text = 'Los Angeles (LAX)'
    plot_type = 'scatterplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

    #  ================== Arcata fittd ================== 
    results_output_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'iter'
    regression_results_dir = results_output_dir + f'/{result_label}_regression_results_smf{weight_text}_{min_channel}_channel_at_least'
    snr_threshold = 20
    M_threshold = [2, 10]
    vmax = [2, 2] # for P and S
    region_text = 'Arcata'
    plot_type = 'scatterplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text)

else:
    #  ================== Arcata transfered random test ================== 
    N_event_fit_list = range(2, 13)
    N_test = 10
    for N_event_fit in N_event_fit_list:
        for i_test in range(N_test):

            results_output_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate'
            peak_file_name = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
            result_label = 'transfer'
            regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
            snr_threshold = 5
            vmax = 50
            M_threshold = [2, 10]
            vmax = [2, 2] # for P and S
            region_text = 'Transfered scaling for Arcata'
            plot_type = 'scatterplot'

            M_threshold_list.append(M_threshold)
            results_output_dir_list.append(results_output_dir)
            regression_results_dir_list.append(regression_results_dir)
            peak_file_name_list.append(peak_file_name)
            result_label_list.append(result_label)
            snr_threshold_list.append(snr_threshold)
            vmax_list.append(vmax)
            region_text_list.append(region_text) 

    #  ================== Arcata transfered specified test ================== 
    results_output_dir = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Arcata/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'transfer'
    regression_results_dir = results_output_dir + f'/transfer_regression_specified_smf{weight_text}_{min_channel}_channel_at_least/'
    snr_threshold = 10
    vmax = 50
    M_threshold = [2, 10]
    vmax = [2, 2] # for P and S
    region_text = 'Transfered scaling for Arcata'
    plot_type = 'scatterplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text) 

    #  ================== Curie transfered specified test ================== 
    results_output_dir = '/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate'
    peak_file_name = '/kuafu/yinjx/Curie/peak_amplitude_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
    result_label = 'transfer'
    regression_results_dir = results_output_dir + f'/transfer_regression_specified_smf{weight_text}_{min_channel}_channel_at_least_9007/'
    snr_threshold = 5
    vmax = 50
    M_threshold = [2, 10]
    vmax = [2, 2] # for P and S
    region_text = 'Transfered scaling for Curie'
    plot_type = 'scatterplot'

    M_threshold_list.append(M_threshold)
    results_output_dir_list.append(results_output_dir)
    regression_results_dir_list.append(regression_results_dir)
    peak_file_name_list.append(peak_file_name)
    result_label_list.append(result_label)
    snr_threshold_list.append(snr_threshold)
    vmax_list.append(vmax)
    region_text_list.append(region_text) 

    #  ================== Sanriku transfered test ================== 
    N_event_fit_list = range(2, 10)
    N_test = 50
    for N_event_fit in N_event_fit_list:
        for i_test in range(N_test):

            results_output_dir = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate'
            peak_file_name = '/kuafu/yinjx/Sanriku/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
            result_label = 'transfer'
            regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
            snr_threshold = 5
            vmax = 50
            M_threshold = [0, 10]
            vmax = [2, 2] # for P and S
            region_text = 'Transfered scaling for Sanriku'
            plot_type = 'scatterplot'

            M_threshold_list.append(M_threshold)
            results_output_dir_list.append(results_output_dir)
            regression_results_dir_list.append(regression_results_dir)
            peak_file_name_list.append(peak_file_name)
            result_label_list.append(result_label)
            snr_threshold_list.append(snr_threshold)
            vmax_list.append(vmax)
            region_text_list.append(region_text) 
            
    #  ================== LAX transfered test ================== 
    N_event_fit_list = range(2, 10)
    N_test = 50
    for N_event_fit in N_event_fit_list:
        for i_test in range(20, N_test):

            results_output_dir = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate'
            peak_file_name = '/kuafu/yinjx/LA_Google/peak_ampliutde_scaling_results_strain_rate/peak_amplitude_events/calibrated_peak_amplitude.csv'
            result_label = 'transfer'
            regression_results_dir = results_output_dir + f'/transfer_regression_test_smf{weight_text}_{min_channel}_channel_at_least/{N_event_fit}_fit_events_{i_test}th_test'
            snr_threshold = 10
            vmax = 50
            M_threshold = [0, 10]
            vmax = [2, 2] # for P and S
            region_text = 'Transfered scaling for LAX'
            plot_type = 'scatterplot'

            M_threshold_list.append(M_threshold)
            results_output_dir_list.append(results_output_dir)
            regression_results_dir_list.append(regression_results_dir)
            peak_file_name_list.append(peak_file_name)
            result_label_list.append(result_label)
            snr_threshold_list.append(snr_threshold)
            vmax_list.append(vmax)
            region_text_list.append(region_text)



#%%
for ii in range(len(peak_file_name_list)):
    
    peak_file_name = peak_file_name_list[ii]
    regression_results_dir = regression_results_dir_list[ii]
    result_label = result_label_list[ii]
    snr_threshold = snr_threshold_list[ii]
    M_threshold = M_threshold_list[ii]
    vmax = vmax_list[ii]
    region_text = region_text_list[ii]
    print(regression_results_dir)

    # load results
    peak_amplitude_df = pd.read_csv(peak_file_name)
    peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
    peak_amplitude_df = filter_event_first_order(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)
    if 'Arcata' in results_output_dir:
        good_events_list = [73736021, 73739276, 73747016, 73751651, 73757961, 73758756, 
            73735891, 73741131, 73747621, 73747806, 73748011, 73753546, 73743421]
        peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.event_id.isin(good_events_list)]
        good_channel_list = np.concatenate([np.arange(0, 2750), np.arange(3450, 10000)])
        peak_amplitude_df = filter_event_first_order(peak_amplitude_df, channel_list=good_channel_list)

        event_id_fit_P0 = [73736021, 73747016, 73735891, 73747621, 73747806, 73748011, 73739346, 73741131, 73743421] #[73736021, 73747016, 73747621] 
        peak_amplitude_df_P = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit_P0)]
        event_id_fit_S0 = [73736021, 73747016, 73735891, 73747621, 73747806, 73748011, 73739346, 73741131, 73743421] 
        peak_amplitude_df_S = peak_amplitude_df[peak_amplitude_df.event_id.isin(event_id_fit_S0)]
    
    if 'Sanriku' in results_output_dir: # some special processing for Sanriku data
        peak_amplitude_df = peak_amplitude_df[peak_amplitude_df.QA == 'Yes']
        peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 4130].index)
        #peak_amplitude_df = peak_amplitude_df.drop(index=peak_amplitude_df[peak_amplitude_df.event_id == 1580].index)


    site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{result_label}.csv')

    try:
        regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{result_label}.pickle")
        # use the measured peak amplitude to estimate the magnitude
        if 'Arcata' in results_output_dir:
            peak_amplitude_df = peak_amplitude_df_P
        magnitude_P, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regP, site_term_df, wavetype='P')
        final_magnitude_P = get_mean_magnitude(peak_amplitude_df_temp, magnitude_P)
    except:
        print('No P regression results, skip...')
        regP, magnitude_P, peak_amplitude_df_temp, final_magnitude_P = None, None, None, None

    try:
        regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{result_label}.pickle")
        if 'Arcata' in results_output_dir:
            peak_amplitude_df = peak_amplitude_df_S
        magnitude_S, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regS, site_term_df, wavetype='S')
        final_magnitude_S = get_mean_magnitude(peak_amplitude_df_temp, magnitude_S)
    except:
        print('No S regression results, skip...')
        regS, magnitude_S, peak_amplitude_df_temp, final_magnitude_S = None, None, None, None

    # check the STD of magnitude estimation, if too large, discard
    # fig, ax = plt.subplots()
    # ax.hist(final_magnitude_P[~final_magnitude_P.predicted_M_std.isna()].predicted_M_std, label='P')
    # ax.hist(final_magnitude_S[~final_magnitude_S.predicted_M_std.isna()].predicted_M_std, label='S')
    # ax.legend()

    # final_magnitude_P = final_magnitude_P[final_magnitude_P.predicted_M_std < 0.75]
    # final_magnitude_S = final_magnitude_S[final_magnitude_S.predicted_M_std < 0.75]

    # plot figures of strain rate validation
    fig_dir = regression_results_dir + '/figures'
    mkdir(fig_dir)

    xy_lim, height, space = [-1, 8], 8, 0.3

    if result_label == 'iter': 
        try:
            gP = plot_magnitude_seaborn(final_magnitude_P, xlim=xy_lim, ylim=xy_lim, vmin=1, vmax=vmax[0], height=height, space=space, type=plot_type)
            gP.ax_joint.text(0, 7, region_text + f', P wave\n{len(final_magnitude_P.dropna())} events')
            gP.savefig(fig_dir + f'/P_magnitude_prediction_rate_{result_label}_seaborn.png')
            gP.savefig(fig_dir + f'/P_magnitude_prediction_rate_{result_label}_seaborn.pdf')
        except:
            print('No P regression results, skip...')

        try:
            gS = plot_magnitude_seaborn(final_magnitude_S, xlim=xy_lim, ylim=xy_lim, vmin=1, vmax=vmax[1], height=height, space=space, type=plot_type)
            gS.ax_joint.text(0, 7, region_text + f', S wave\n{len(final_magnitude_S.dropna())} events')
            gS.savefig(fig_dir + f'/S_magnitude_prediction_rate_{result_label}_seaborn_x.png')
            gS.savefig(fig_dir + f'/S_magnitude_prediction_rate_{result_label}_seaborn_x.pdf')
        except:
            print('No S regression results, skip...')    

    elif result_label == 'transfer':
        temp = np.load(regression_results_dir + '/transfer_event_list.npz')
        event_id_fit_P = temp['event_id_fit']
        event_id_fit_S = temp['event_id_fit']
        event_id_predict = temp['event_id_predict']

        try:
            final_magnitude_P_fit = final_magnitude_P[final_magnitude_P.event_id.isin(event_id_fit_P)]
            final_magnitude_P_predict = final_magnitude_P[final_magnitude_P.event_id.isin(event_id_predict)]

            gP = plot_magnitude_seaborn(final_magnitude_P_predict, xlim=xy_lim, ylim=xy_lim, vmin=1, vmax=vmax[0], height=height, space=space, type=plot_type)
            gP.ax_joint.plot(final_magnitude_P_fit.magnitude, final_magnitude_P_fit.predicted_M, 'ro')
            # gP.ax_joint.plot(final_magnitude_P_predict.magnitude, final_magnitude_P_predict.predicted_M, 'bo')
            gP.ax_joint.text(-0.5, 7, region_text + f', P wave\n{len(final_magnitude_P_fit.dropna())} events to fit, {len(final_magnitude_P_predict.dropna())} events to predict', fontsize=16)
            gP.savefig(fig_dir + f'/P_magnitude_prediction_rate_{result_label}_seaborn.png')
            gP.savefig(fig_dir + f'/P_magnitude_prediction_rate_{result_label}_seaborn.pdf')
        except:
            print('No valid P wave regression results, skip ...')
            pass

        try:
            final_magnitude_S_fit = final_magnitude_S[final_magnitude_S.event_id.isin(event_id_fit_S)]
            final_magnitude_S_predict = final_magnitude_S[final_magnitude_S.event_id.isin(event_id_predict)]

            gS = plot_magnitude_seaborn(final_magnitude_S_predict, xlim=xy_lim, ylim=xy_lim, vmin=1, vmax=vmax[1], height=height, space=space, type=plot_type)
            gS.ax_joint.plot(final_magnitude_S_fit.magnitude, final_magnitude_S_fit.predicted_M, 'ro')
            # gS.ax_joint.plot(final_magnitude_S_predict.magnitude, final_magnitude_S_predict.predicted_M, 'bo')
            gS.ax_joint.text(-0.5, 7, region_text + f', S wave\n{len(final_magnitude_S_fit.dropna())} events to fit, {len(final_magnitude_S_predict.dropna())} events to predict', fontsize=16)
            gS.savefig(fig_dir + f'/S_magnitude_prediction_rate_{result_label}_seaborn.png')
            gS.savefig(fig_dir + f'/S_magnitude_prediction_rate_{result_label}_seaborn.pdf')
        except:
            print('No valid S wave regression results, skip ...')
            pass

    plt.close('all')

# %%
