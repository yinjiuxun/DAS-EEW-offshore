#%% import modules
import pandas as pd
import numpy as np
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.processing import filter_event_first_order
from utility.regression import predict_magnitude, get_mean_magnitude

def combine_results(peak_amplitude_df_temp, magnitude_X, wave_type):
    wave_type = wave_type.upper()
    individual_magnitude = peak_amplitude_df_temp[['event_id', 'channel_id', 'magnitude', f'snr{wave_type}']]
    individual_magnitude = individual_magnitude.rename(columns={f'snr{wave_type}': 'snr'})
    individual_magnitude['predicted_magnitude'] = magnitude_X
    individual_magnitude['magnitude_error'] = individual_magnitude['predicted_magnitude'] - individual_magnitude['magnitude']
    return individual_magnitude

#%%
#  ================== transfered specified test ================== 
results_output_dir = '../results'
peak_file_name = '../data_files/peak_amplitude/calibrated_peak_amplitude.csv'
result_label = 'transfer'
regression_results_dir = '../results/transfer_regression_with_9007/'
fig_dir = regression_results_dir + '/figures'
mkdir(fig_dir)

# some parameters
snr_threshold = 10
M_threshold = [2, 10]
min_channel = 100 # do regression only on events recorded by at least 100 channels
region_text = 'Transfered scaling for Curie'

#%%
# load peak amplitude
peak_amplitude_df = pd.read_csv(peak_file_name)
peak_amplitude_df['distance_in_km'] = peak_amplitude_df['calibrated_distance_in_km']
peak_amplitude_df = filter_event_first_order(peak_amplitude_df, M_threshold=M_threshold, snr_threshold=snr_threshold, min_channel=min_channel)

# load calibrated site terms
site_term_df = pd.read_csv(regression_results_dir + f'/site_terms_{result_label}.csv')

# load events 
temp = np.load(regression_results_dir + '/transfer_event_list.npz')
event_id_fit_P = temp['event_id_fit_P']
event_id_fit_S = temp['event_id_fit_S']
event_id_predict = temp['event_id_predict']

try:
    regP = sm.load(regression_results_dir + f"/P_regression_combined_site_terms_{result_label}.pickle")
    # use the measured peak amplitude to estimate the magnitude
    magnitude_P, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regP, site_term_df, wavetype='P')
    individual_magnitude_P = combine_results(peak_amplitude_df_temp, magnitude_P, 'P')
    final_magnitude_P = get_mean_magnitude(peak_amplitude_df_temp, magnitude_P)

    final_magnitude_P_fit = final_magnitude_P[final_magnitude_P.event_id.isin(event_id_fit_P)]
    final_magnitude_P_predict = final_magnitude_P[final_magnitude_P.event_id.isin(event_id_predict)]

except:
    print('No P regression results, skip...')
    regP, magnitude_P, peak_amplitude_df_temp, final_magnitude_P = None, None, None, None

try:
    regS = sm.load(regression_results_dir + f"/S_regression_combined_site_terms_{result_label}.pickle")
    magnitude_S, peak_amplitude_df_temp = predict_magnitude(peak_amplitude_df, regS, site_term_df, wavetype='S')
    individual_magnitude_S = combine_results(peak_amplitude_df_temp, magnitude_S, 'S')
    final_magnitude_S = get_mean_magnitude(peak_amplitude_df_temp, magnitude_S)

    final_magnitude_S_fit = final_magnitude_S[final_magnitude_S.event_id.isin(event_id_fit_S)]
    final_magnitude_S_predict = final_magnitude_S[final_magnitude_S.event_id.isin(event_id_predict)]
    
except:
    print('No S regression results, skip...')
    regS, magnitude_S, peak_amplitude_df_temp, final_magnitude_S = None, None, None, None

# %%
# Plot the calibrated site terms
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(site_term_df.channel_id, site_term_df.site_term_P, '.', label='P')
ax.plot(site_term_df.channel_id, site_term_df.site_term_S, '.', label='S')
ax.legend(loc=4)
ax.set_xlabel('Channel number')
ax.set_ylabel('Site term (log10)')
plt.savefig(fig_dir + f'/site_terms_{result_label}.png', bbox_inches='tight')

#%%
# plot the magnitude estimation results
fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
gca = ax[0]
sns.violinplot(data=individual_magnitude_P, x="event_id", y="magnitude_error", scale='count', bw=0.1, width=1, ax=gca)
gca.set_ylabel('Magnitude P error')
gca.plot([-1, 3], [0, 0], '-k', zorder=-1)
gca.text(-0.7, 1.15, f'Catalog M: {final_magnitude_P_predict.iloc[0, 1]}\nEstimated M: {final_magnitude_P_predict.iloc[0, 2]:.1f}',fontsize=15)
gca.text(0.3,  1.15, f'Catalog M: {final_magnitude_P_predict.iloc[1, 1]}\nEstimated M: {final_magnitude_P_predict.iloc[1, 2]:.1f}',fontsize=15)
gca.set_xlabel('')


event_names = ['60 km W of La Ligua', '40 km NW of Valparaíso', '']
gca = ax[1]
sns.violinplot(data=individual_magnitude_S, x="event_id", y="magnitude_error", scale='count',  bw=0.1, ax=gca)
gca.set_ylabel('Magnitude S error')
gca.plot([-1, 3], [0, 0], '-k', zorder=-1)
gca.text(-0.7, 1.15, f'Catalog M: {final_magnitude_S_predict.iloc[0, 1]}\nEstimated M: {final_magnitude_S_predict.iloc[0, 2]:.1f}',fontsize=15)
gca.text(0.3,  1.15, f'Catalog M: {final_magnitude_S_predict.iloc[1, 1]}\nEstimated M: {final_magnitude_S_predict.iloc[1, 2]:.1f}',fontsize=15)
gca.set_xlim(-1, 2)
gca.set_xlabel('')
gca.set_xticklabels(event_names, rotation=15)

plt.savefig(fig_dir + f'/magnitude_error_violin_{result_label}.png', bbox_inches='tight')
