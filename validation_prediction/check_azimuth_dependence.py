#%%
# Import modules
import pandas as pd
#from sep_util import read_file
import numpy as np
import tqdm
import glob
import psutil 
Ncores = psutil.cpu_count(logical = False) # Maximum number of cores that can be employed

import warnings
from obspy.geodetics import locations2degrees

# Plotting
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
# load modules
from utility.processing import calculate_SNR
from utility.loading import load_event_data, load_phasenet_pick
from utility.general import *

# %matplotlib inline
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
    'savefig.facecolor': 'white'
}
matplotlib.rcParams.update(params)

#%%
results_output_dir = '/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM'
peak_file_name = 'peak_amplitude_multiple_arrays.csv'
peak_amplitude_df = pd.read_csv(results_output_dir + f'/{peak_file_name}')

region_folder_list = ['/kuafu/EventData/Ridgecrest', 
                     '/kuafu/EventData/Mammoth_south',
                     '/kuafu/EventData/Mammoth_north']

catalog_list, das_info_list = [], []
for i_region in range(len(region_folder_list)):
    catalog_list.append(pd.read_csv(region_folder_list[i_region] + '/catalog.csv'))
    das_info_list.append(pd.read_csv(region_folder_list[i_region] + '/das_info.csv'))

# %%
def calculate_array_direction(latitudes, longitudes):
    """Calculate the azimuth angle along the DAS array"""
    # Convert latitude and longitude to radians
    lat1 = np.radians(latitudes[:-1])
    lat2 = np.radians(latitudes[1:])
    lon1 = np.radians(longitudes[:-1])
    lon2 = np.radians(longitudes[1:])

    # Calculate differences between latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate azimuth
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    azimuth = np.degrees(np.arctan2(y, x))
    
    # Normalize to 0-360 degrees
    azimuth[azimuth < 0] += 360

    return azimuth

def calculate_azimuth(lat1, lon1, lat2, lon2):
    """Calculate the azimuth angle in degrees between two points on the Earth's surface given their latitude and longitude coordinates."""
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lon_rad = np.radians(lon2 - lon1)

    # Calculate the azimuth angle in radians
    y = np.sin(delta_lon_rad) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon_rad)
    azimuth_rad = np.arctan2(y, x)

    # Convert to degrees
    azimuth_deg = np.degrees(azimuth_rad)

    # Normalize to 0-360 degrees
    azimuth_deg[azimuth_deg < 0] += 360

    return azimuth_deg
# %%
# show the azimuthal dependence
def normalize_peak(peak_amplitude_azimuth_pd, wave_type, percentile=0.5):
    wave_type = wave_type.upper()
    peak_amplitude_azimuth_pd_normalized = peak_amplitude_azimuth_pd.copy()
    if wave_type == 'P':
        ii_normal_factor_P = abs(peak_amplitude_azimuth_pd.peak_P - peak_amplitude_azimuth_pd.peak_P.quantile(percentile)).argmin()
        ii_normal_factor_P = abs(peak_amplitude_azimuth_pd.peak_P.iloc[ii_normal_factor_P])#/np.cos(peak_amplitude_azimuth_pd.azimuth.iloc[ii_normal_factor_P]/np.pi/2))**2
        
        # normalized peak amplitude
        peak_amplitude_azimuth_pd_normalized.peak_P = peak_amplitude_azimuth_pd_normalized.peak_P/ii_normal_factor_P
    elif wave_type == 'S':
        ii_normal_factor_S = abs(peak_amplitude_azimuth_pd.peak_S - peak_amplitude_azimuth_pd.peak_S.quantile(percentile)).argmin()
        ii_normal_factor_S = abs(peak_amplitude_azimuth_pd.peak_S.iloc[ii_normal_factor_S])#/np.sin(2*peak_amplitude_azimuth_pd.azimuth.iloc[ii_normal_factor_S]/np.pi/2)*2)
        
        # normalized peak amplitude
        peak_amplitude_azimuth_pd_normalized.peak_S = peak_amplitude_azimuth_pd_normalized.peak_S/ii_normal_factor_S
    else:
        raise ValueError('wave_type needs to be either "P" or "S"!')
    return peak_amplitude_azimuth_pd_normalized

region_list = peak_amplitude_df.region.unique()
for i_region, region in enumerate(['mammothN']):#enumerate(region_list):
    i_region = 2
    print(region)
    
    catalog = catalog_list[i_region]
    das_info = das_info_list[i_region]
    event_id_list = peak_amplitude_df[peak_amplitude_df.region == region].event_id.unique()
    azimuth_das = calculate_array_direction(np.array(das_info.latitude), np.array(das_info.longitude))
    azimuth_das = np.concatenate([azimuth_das, azimuth_das[-1, np.newaxis]])
    plt.close('all')
    fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(12, 6))

    for event_id in event_id_list[::1]:
        lat1 = das_info.latitude
        lon1 = das_info.longitude
        lat2 = catalog[catalog.event_id == event_id].latitude.iloc[0]
        lon2 = catalog[catalog.event_id == event_id].longitude.iloc[0]
        azimuth_deg = calculate_azimuth(lat1, lon1, lat2, lon2) - azimuth_das

        # plt.plot(lon1, lat1, 'k.')
        # plt.plot(lon2, lat2, 'rp')

        azimuth_pd = pd.DataFrame()
        azimuth_pd['channel_id'] = das_info['index'].astype('float')
        azimuth_pd['azimuth'] = azimuth_deg

        peak_amplitude_df_event = peak_amplitude_df[peak_amplitude_df.event_id == event_id][['channel_id', 'peak_P', 'peak_S']]
        peak_amplitude_azimuth_pd = pd.merge(left=peak_amplitude_df_event, right=azimuth_pd, left_on='channel_id', right_on='channel_id')

        peak_amplitude_P_normalized = normalize_peak(peak_amplitude_azimuth_pd, 'P', 0.8)
        peak_amplitude_S_normalized = normalize_peak(peak_amplitude_azimuth_pd, 'S', 0.8)
        
        # show azimuthal variation
        theta_theory = np.linspace(0, np.pi*2.1, 100)
        r_P_theory = abs(np.cos(theta_theory)**2)
        r_S_theory = abs(np.sin(theta_theory*2)/2)

        theta = peak_amplitude_azimuth_pd.azimuth/2/np.pi
        r_P = peak_amplitude_P_normalized.peak_P
        r_S = peak_amplitude_S_normalized.peak_S
        
        ax[0].plot(theta, r_P, 'b.', alpha=0.01)
        ax[0].plot(theta_theory, r_P_theory, '-k', alpha=1)
        ax[0].set_rlim(0, 3.6)
        ax[0].set_title('Normalized peak P amplitude')

        ax[1].plot(theta, r_S, 'b.', alpha=0.01)
        ax[1].plot(theta_theory, r_S_theory, '-k', alpha=1)
        ax[1].set_rlim(0, 3.6)
        ax[1].set_title('Normalized peak S amplitude')

    plt.savefig(f'/kuafu/yinjx/multi_array_combined_scaling/combined_strain_scaling_RM/data_figures/azimuth_dependence_{region}.png')

# %%
