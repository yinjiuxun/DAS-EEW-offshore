#%% import modules
import os
import pandas as pd
#from sep_util import read_file
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

import sys
sys.path.append('../')
from utility.general import *
from utility.loading import load_event_data
from scipy.interpolate import griddata
from scipy.signal import find_peaks

from scipy.interpolate import griddata
from obspy.geodetics import locations2degrees
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# set plotting parameters 
fontsize=18
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 100,  # to adjust notebook inline plot size
    'axes.labelsize': fontsize, # fontsize for x and y labels (was 10)
    'axes.titlesize': fontsize,
    'font.size': fontsize,
    'legend.fontsize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'text.usetex':False,
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white'
}
mpl.rcParams.update(params)

def convert_info_from_peak(DAS_lon, DAS_lat, eq_info, lon_grid, lat_grid, ii_max):
    """A convenient function to extract information"""
    opt_lat, opt_lon = lat_grid.flatten()[ii_max], lon_grid.flatten()[ii_max]
    distance_to_source = locations2degrees(DAS_lat, DAS_lon, opt_lat, opt_lon) * 113 # in km
    distance_to_source = np.sqrt(eq_info.iloc[0, :].depth_km**2 + distance_to_source**2)
    return opt_lat,opt_lon,distance_to_source
#%% 
# 1. Specify earthquake to look at
# load event waveforms
region_label = 'curie' 
time_step = 50 # time step to measure peaks

tt_dir = '../data_files/pickings/theoretical_arrival_time0' ##
test_event_id =  9006 #9001, 9006
show_loc = 3

if test_event_id == 9006:
    event_name = 'Valparaiso'
elif test_event_id == 9001:
    event_name = 'LaLigua'
else:
    raise NameError

tt_shift_p, tt_shift_s = 0, 0
given_range_P = None
given_range_S = None
every_n_channel = 50

catalog = pd.read_csv('../data_files/catalogs/catalog.csv')
das_waveform_path = '../data_files/event_data'

#%%
# 2. Specify the array to look at
# load the DAS array information
DAS_info = pd.read_csv('../data_files/das_info/das_info.csv')
# Only keep the first 7000 channels
DAS_info = DAS_info[DAS_info['index'] < 7000]

ML_picking_dir = '../data_files/pickings/picks_phasenet_das'
use_ML_picking = True

#%%
# 3. Specify the coefficients to load
fig_output_dir = '../results/real_time_test_downsampled'
mkdir(fig_output_dir)
fig_output_dir += f'/{test_event_id}_downsample_{every_n_channel}'
mkdir(fig_output_dir)

# Load DAS channels
DAS_channel_num = DAS_info.shape[0]
DAS_index = DAS_info['index']
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

# Load the DAS data
strain_rate_temp, info = load_event_data(das_waveform_path, test_event_id)
strain_rate_temp = strain_rate_temp[:, DAS_index]
das_dt = info['dt_s']
nt = strain_rate_temp.shape[0]
das_time0 = np.arange(nt) * das_dt-30

# Load regression results
regP = sm.load("../results/transfer_regression_with_9007/P_regression_combined_site_terms_transfer.pickle")
regS = sm.load("../results/transfer_regression_with_9007/S_regression_combined_site_terms_transfer.pickle")
site_terms_df = pd.read_csv("../results/transfer_regression_with_9007/site_terms_transfer.csv")
site_terms_df = site_terms_df[site_terms_df.channel_id<=DAS_index.max()]
DAS_channel_num = len(DAS_info)

#%%
# have the site term in the same shape of original data, fill nan instead
channel_id = np.array(site_terms_df[site_terms_df.region == region_label]['channel_id'].astype('int'))
site_terms_P = np.zeros(shape=(1, DAS_channel_num)) * np.nan
site_terms_S = np.zeros(shape=(1, DAS_channel_num)) * np.nan

site_terms_P[:, channel_id] = site_terms_df[site_terms_df.region == region_label]['site_term_P']
site_terms_S[:, channel_id] = site_terms_df[site_terms_df.region == region_label]['site_term_S']

site_terms_P = site_terms_P[:, np.array(DAS_index)-1]
site_terms_S = site_terms_S[:, np.array(DAS_index)-1]


#%% only keep every X channels to mimic the channel downsampling
ii_channel = range(0, DAS_index.shape[0], every_n_channel)
strain_rate= strain_rate_temp[:, ii_channel]
site_terms_P = site_terms_P[:, ii_channel]
site_terms_S = site_terms_S[:, ii_channel]
DAS_lon = DAS_lon[ii_channel]
DAS_lat = DAS_lat[ii_channel]

#%%
# load the travel time and process
def remove_ml_tt_outliers(ML_picking, das_dt, tdiff=10, given_range=None):
    temp = ML_picking.drop(index=ML_picking[abs(ML_picking.phase_index - ML_picking.phase_index.median())*das_dt >= tdiff].index)
    if given_range:
        try:
            temp = temp[(temp.phase_index>=given_range[0]/das_dt) & (temp.phase_index<=given_range[1]/das_dt)]
        except:
            print('cannot specify range, skip...')
    return temp

eq_info = catalog[catalog.event_id == test_event_id]
eq_lat = eq_info.latitude # lat
eq_lon = eq_info.longitude # lon
eq_mag = eq_info.magnitude # catalog magnitude
eq_time = eq_info.event_time # eq time
eq_depth = eq_info.depth_km  # eq depth

# get distance from earthquake to each channel
distance_to_source0 = locations2degrees(DAS_lat, DAS_lon, eq_lat, eq_lon) * 113 # in km
distance_to_source0 = np.sqrt(eq_info.iloc[0, :].depth_km**2 + distance_to_source0**2)
distance_to_source0 = distance_to_source0[np.newaxis, :]

# Use the picking from PhaseNet-DAS
tt_tp = np.zeros(shape=(1, DAS_channel_num))*np.nan
tt_ts = tt_tp.copy()

ML_picking = pd.read_csv(ML_picking_dir + f'/{test_event_id}.csv')

ML_picking_P = ML_picking[ML_picking.phase_type == 'P']
ML_picking_S = ML_picking[ML_picking.phase_type == 'S']
ML_picking_P = remove_ml_tt_outliers(ML_picking_P, das_dt, tdiff=5, given_range=given_range_P)
ML_picking_S = remove_ml_tt_outliers(ML_picking_S, das_dt, tdiff=20, given_range=given_range_S)

ML_picking_P = ML_picking_P[ML_picking_P.channel_index<=DAS_index.max()]
ML_picking_S = ML_picking_S[ML_picking_S.channel_index<=DAS_index.max()]

tt_tp[0, ML_picking_P.channel_index] = das_time0[ML_picking_P.phase_index]
tt_ts[0, ML_picking_S.channel_index] = das_time0[ML_picking_S.phase_index]

DAS_channel_num = len(ii_channel)
tt_tp = tt_tp[:, ii_channel]
tt_ts = tt_ts[:, ii_channel]

# Some DUMP process to handle the arrival time
tt_tp_temp = tt_tp.copy()
tt_tp_temp[np.isnan(tt_tp_temp)]=1e10
tt_ts_temp = tt_ts.copy()
tt_ts_temp[np.isnan(tt_ts_temp)]=1e10


#%% 
# Estimate the response time for both DAS and permanent seimsic stations
# Load nearby stations
stations = pd.read_csv('../data_files/nearby_stations/curie_nearby_stations.csv', index_col=None)
STA_lon = stations.Longitude
STA_lat = stations.Latitude
# get distance from earthquake to all the regional stations
distance_to_source_sta = locations2degrees(STA_lat, STA_lon, eq_lat, eq_lon) 

travel_time_table_file = '../data_files/travel_time_table/travel_time_table.npz'
temp = np.load(travel_time_table_file)
distance_grid = temp['distance_grid']
depth_grid = temp['depth_grid']
tavel_time_p_grid = temp['tavel_time_p_grid']
tavel_time_s_grid = temp['tavel_time_s_grid']

P_arrival_STA = griddata(np.array([distance_grid.flatten(), depth_grid.flatten()]).T, tavel_time_p_grid.flatten(), (distance_to_source_sta, np.ones(distance_to_source_sta.shape)*eq_depth.iloc[0]))
S_arrival_STA = griddata(np.array([distance_grid.flatten(), depth_grid.flatten()]).T, tavel_time_s_grid.flatten(), (distance_to_source_sta, np.ones(distance_to_source_sta.shape)*eq_depth.iloc[0]))

# estimate the time when 4 station recorded P
sorted_P_arrival = np.sort(P_arrival_STA)
n_sta = 4
SEIS_response_time = sorted_P_arrival[n_sta-1]
# estimate the DAS response time when the S is recorded
DAS_response_time = np.nanmin(tt_ts)

#%%
#extract peak amplitude based on the phase arrival time
strain_rate_clipped_P = strain_rate.copy()
das_time1 = das_time0[:, np.newaxis]
strain_rate_clipped_P[das_time1<=tt_tp_temp]=np.nan
strain_rate_clipped_P[(das_time1>tt_tp_temp+2)] = np.nan

strain_rate_clipped_S = strain_rate.copy()
strain_rate_clipped_S[(das_time1<tt_ts_temp)] = np.nan
strain_rate_clipped_S[(das_time1>tt_ts_temp+2)] = np.nan

# matrices to store measured P and S peak amplitude
data_peak_mat_P = np.zeros((np.ceil(strain_rate.shape[0]/time_step).astype('int'), strain_rate.shape[1]))
data_peak_mat_S = np.zeros((np.ceil(strain_rate.shape[0]/time_step).astype('int'), strain_rate.shape[1]))

for i, ii_win in enumerate(range(0, strain_rate.shape[0], time_step)):
    # keep the maximum unchanged
    data_peak_mat_P[i, :] = np.nanmax(abs(strain_rate_clipped_P[0:(ii_win+time_step), :]), axis=0)     
    data_peak_mat_S[i, :] = np.nanmax(abs(strain_rate_clipped_S[0:(ii_win+time_step), :]), axis=0)    

das_time = das_time0[::time_step]

#%%
# calculate magnitude for each channel
# replace distance_to_source with distance_mat
mag_estimate_P = (np.log10(data_peak_mat_P+1e-12) - site_terms_P - np.log10(distance_to_source0)*regP.params['np.log10(distance_in_km)'])/regP.params['magnitude']
median_mag_P = np.nanmedian(mag_estimate_P, axis=1)
mean_mag_P = np.nanmean(mag_estimate_P, axis=1)
std_mag_P = np.nanstd(mag_estimate_P, axis=1)

mag_estimate_S = (np.log10(data_peak_mat_S+1e-12) - site_terms_S - np.log10(distance_to_source0)*regS.params['np.log10(distance_in_km)'])/regS.params['magnitude']
median_mag_S = np.nanmedian(mag_estimate_S, axis=1)
mean_mag_S = np.nanmean(mag_estimate_S, axis=1)
std_mag_S = np.nanstd(mag_estimate_S, axis=1)

# combine both P and S to give final magnitude 
das_time_mat = das_time[:, np.newaxis]
mag_estimate_final = mag_estimate_P.copy()
# mag_estimate_final[das_time_mat>=cvm_tt_ts] = mag_estimate_S[das_time_mat>=cvm_tt_ts]
mag_estimate_final = np.nanmedian(np.array([mag_estimate_P, mag_estimate_S]), axis=0)

median_mag = np.nanmedian(mag_estimate_final, axis=1)
mean_mag = np.nanmean(mag_estimate_final, axis=1)
std_mag = np.nanstd(mag_estimate_final, axis=1)

#%% 
# Having all together as one figure
time_rang_show = [0, 120]
if np.isnan(np.nanmin(tt_tp)):
    time_rang_show[0] = np.nanmin(tt_ts)-10
else:
    time_rang_show[0] = np.nanmin(tt_tp)-3
if not np.isnan(np.nanmax(tt_ts)):
    time_rang_show[1] = np.nanmedian(tt_ts)+10

fig, ax = plt.subplots(3, 1, figsize=(10,20))
# Strain
pclip=99
clipVal = np.percentile(np.absolute(strain_rate), pclip)
gca = ax[0]
clb = gca.imshow(strain_rate.T, 
            extent=[das_time0[0], das_time0[-1], mag_estimate_P.shape[1], 0],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'), interpolation='none')

gca.plot(tt_tp.flatten(), np.arange(tt_tp.shape[1]), '--k', linewidth=2, label='P')
gca.plot(tt_ts.flatten(), np.arange(tt_ts.shape[1]), '-k', linewidth=2, label='S')

gca.vlines(x=[np.nanmin(tt_tp), np.nanmin(tt_ts)], ymax=strain_rate.shape[1], ymin=0, label='earliest tP and tS', color='b', linestyle='--')

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.invert_yaxis()

axins1 = inset_axes(gca,
                    width="2%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='lower left')
gca.set_title(f'id: {test_event_id}, magnitude: {eq_mag.values[0]}', fontsize=20)
fig.colorbar(clb, cax=axins1, orientation="vertical", label='strain rate ($10^{-6}$/s)')

# Colormap for Magnitude
cmap = plt.cm.get_cmap('OrRd', 6)

gca = ax[1]
clb = gca.imshow(mag_estimate_final.T, 
            extent=[das_time[0], das_time[-1], mag_estimate_final.shape[1], 0],
            aspect='auto', vmin=0, vmax=6, cmap=cmap, interpolation='none')
gca.plot(tt_tp.flatten(), np.arange(tt_tp.shape[1]), '--k', linewidth=2)
gca.plot(tt_ts.flatten(), np.arange(tt_ts.shape[1]), '-k', linewidth=2)
gca.vlines(x=[np.nanmin(tt_tp), np.nanmin(tt_ts)], ymax=mag_estimate_final.shape[1], ymin=0, label='earliest tP and tS', color='b', zorder=10)

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.invert_yaxis()

axins2 = inset_axes(gca,
                    width="2%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='lower left')
fig.colorbar(clb, cax=axins2, orientation="vertical", label='Magnitude')

gca=ax[2]
gca.plot(das_time, median_mag, '-r', linewidth=4, alpha=1, zorder=3, label='mean')
gca.plot(das_time, median_mag-std_mag, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
gca.plot(das_time, median_mag+std_mag, '--r', linewidth=2, alpha=0.5, zorder=3)
gca.vlines(x=[np.nanmin(tt_tp)], ymax=10, ymin=0, label='earliest tP', color='b', linestyle='--')

# label the response time
gca.vlines(x=[DAS_response_time], ymax=10, ymin=0, label='DAS response time (earliest tS)', color='#36756E', linewidth=4)
gca.vlines(x=[SEIS_response_time], ymax=10, ymin=0, label='Seismic station response time', color='#724819', linewidth=4)

mag_estiamte_P = median_mag[int(np.nanmax(tt_tp+2+30)/das_dt/time_step)]
gca.text(x=np.nanmax(tt_tp+2), y=mag_estiamte_P+0.3, s=f'M {mag_estiamte_P:.2f}', color='r', fontsize=18)
mag_estiamte_S = median_mag[int(np.nanmax(tt_ts+2+30)/das_dt/time_step)]
gca.text(x=np.nanmax(tt_ts+2), y=mag_estiamte_S+0.3, s=f'M {mag_estiamte_S:.2f}', color='r', fontsize=18)

gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')
gca.set_yticks(np.arange(0, 9))
gca.set_ylim(0, eq_mag.values*1.4)
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.set_xlabel('Time (s)')
gca.set_ylabel('Estimated Magnitude')
gca.legend(loc=4, fontsize=10)

plt.subplots_adjust(hspace=0.2)

# some final handling
letter_list = [str(chr(k+97)) for k in range(0, 10)]
k=0
for ii in range(0, 3):
    ax[ii].annotate(f'({letter_list[k]})', xy=(-0.05, 1.0), xycoords=ax[ii].transAxes, fontsize=20)
    k+=1

plt.savefig(fig_output_dir + f'/{region_label}_{event_name}_estimated_mag_image_true_location.png', bbox_inches='tight')




#%%
# real time locating and estimating magnitude
# match the arrival time
from numpy.linalg import norm

def match_arrival(event_arrival_observed, event_arrival_template, misfit_type='l1', demean=True):
    ii_nan = np.isnan(event_arrival_observed)
    event_arrival = event_arrival_observed[~ii_nan, np.newaxis]
    template = event_arrival_template[~ii_nan, :]

    if demean:
        # remove mean
        event_arrival = event_arrival - np.mean(event_arrival, axis=0, keepdims=True)
        template = template - np.mean(template, axis=0, keepdims=True)

    if misfit_type == 'l1':
        norm_diff = np.nanmean(abs(event_arrival - template), axis=0) # L1 norm
    elif misfit_type == 'l2':
        norm_diff = np.sqrt(np.nanmean((event_arrival - template)**2, axis=0)) # L2 norm
        
    ii_min = np.nanargmin(norm_diff)

    return ii_min, norm_diff

def misfit_to_probability(misfit):
    probability = 1/misfit/np.nansum(1/misfit)

    return probability

#%%
# Load the calculated templates
arrival_time_curve_template_file =  '../data_files/travel_time_table/arrival_time_template.npz'
temp = np.load(arrival_time_curve_template_file)
distance_to_source_all = temp['distance_grid']
mean_distance = np.nanmean(distance_to_source_all, axis=0) # average distance from each grid point to an array
# event_arrival_P_template = temp['event_arrival_P'][:(DAS_index.max()+1), :]
# event_arrival_S_template = temp['event_arrival_S'][:(DAS_index.max()+1), :]
event_arrival_P_template = temp['event_arrival_P'][:(DAS_index.max()+1), :][ii_channel, :]
event_arrival_S_template = temp['event_arrival_S'][:(DAS_index.max()+1), :][ii_channel, :]
event_SP_diff_template = event_arrival_S_template - event_arrival_P_template
lon_grid = temp['lon_grid']
lat_grid = temp['lat_grid']

assumed_distance = 0.5
# matrices to store the distance 
location_mat = np.zeros((len(das_time), 2))*np.nan
location_mat1, location_mat2 = location_mat.copy(), location_mat.copy()

distance_mat = np.zeros((len(das_time), strain_rate.shape[1]))*np.nan
distance_mat1 = np.zeros((len(das_time), strain_rate.shape[1]))*np.nan
distance_mat2 = np.zeros((len(das_time), strain_rate.shape[1]))*np.nan

probability_mat = np.zeros((len(das_time), event_arrival_S_template.shape[1]))*np.nan

ttp = tt_tp_temp.squeeze()
tts = tt_ts_temp.squeeze()

distance_error_time = np.zeros((len(das_time), 2))*np.nan

for i_time in range(len(das_time)):
    print(f'{das_time[i_time]:.2f} s')
    P_obs_arrival = np.zeros(ttp.shape)*np.nan
    S_obs_arrival = np.zeros(tts.shape)*np.nan
    SP_diff_obs_arrival = np.zeros(tts.shape)*np.nan
    iip = ttp<=das_time[i_time]
    if len(ttp[iip]) == 0:
        distance_mat[i_time, :] = np.nan
        continue
    else:
        P_obs_arrival[iip] = ttp[iip] 
        _, norm_diff_P = match_arrival(P_obs_arrival, event_arrival_P_template, misfit_type='l1')
        probability = misfit_to_probability(norm_diff_P)

        iis = tts<=das_time[i_time]
        if len(tts[iis]) == 0: # before S comes
            xx = np.where(mean_distance<=assumed_distance)[0]
            ii_x = np.nanargmax(probability[xx])
            ii_max = xx[ii_x]

            ii_max1, ii_max2 = ii_max, ii_max

        else:# len(tts[iis]) > 0 when S comes
            S_obs_arrival[iis] = tts[iis] 
            SP_diff_obs_arrival = S_obs_arrival - P_obs_arrival

            if len(SP_diff_obs_arrival[~np.isnan(SP_diff_obs_arrival)])>0:
                _, norm_diff_SP = match_arrival(SP_diff_obs_arrival, event_SP_diff_template, misfit_type='l1', demean=False)
                probability = probability * misfit_to_probability(norm_diff_SP)
                ii_max = np.nanargmax(probability)

                peaks, _ = find_peaks(probability, distance=500, height=np.nanmax(probability)*0.6)
                ii_max1 = peaks[0]
                ii_max2 = peaks[1]

    probability_mat[i_time, :] = probability

    opt_lat, opt_lon, distance_to_source = convert_info_from_peak(DAS_lon, DAS_lat, eq_info, lon_grid, lat_grid, ii_max)

    # put into matrices
    location_mat[i_time, 0] = opt_lat
    location_mat[i_time, 1] = opt_lon
    distance_mat[i_time, :] = distance_to_source

    # opt_lat1, opt_lon1 = lat_grid.flatten()[ii_max1], lon_grid.flatten()[ii_max1]
    # opt_lat2, opt_lon2 = lat_grid.flatten()[ii_max2], lon_grid.flatten()[ii_max2]

    opt_lat1, opt_lon1, distance_to_source1 = convert_info_from_peak(DAS_lon, DAS_lat, eq_info, lon_grid, lat_grid, ii_max1)
    opt_lat2, opt_lon2, distance_to_source2 = convert_info_from_peak(DAS_lon, DAS_lat, eq_info, lon_grid, lat_grid, ii_max2)
    
    location_mat1[i_time, 0], location_mat1[i_time, 1], distance_mat1[i_time, :] = opt_lat1, opt_lon1, distance_to_source1
    location_mat2[i_time, 0], location_mat2[i_time, 1], distance_mat2[i_time, :] = opt_lat2, opt_lon2, distance_to_source2
    
    distance_error_time[i_time, 0] = locations2degrees(eq_lat, eq_lon, opt_lat1, opt_lon1) * 113
    distance_error_time[i_time, 1] = locations2degrees(eq_lat, eq_lon, opt_lat2, opt_lon2) * 113

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(das_time, distance_error_time[:, 0], label='Location 1')
ax.plot(das_time, distance_error_time[:, 1], label='Location 2')
ax.legend()
ax.set_title(event_name + f' M{eq_mag.values[0]:.2f}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Distance to epicenter (km)')
plt.savefig(fig_output_dir + f'/{region_label}_{event_name}_location_error_time.png', bbox_inches='tight')

#%%
# calculate magnitude for each channel
# replace distance_to_source with distance_mat
def calculate_magnitude(regP, regS, site_terms_P, site_terms_S, data_peak_mat_P, data_peak_mat_S, distance_mat):
    mag_estimate_P = (np.log10(data_peak_mat_P+1e-12) - site_terms_P - np.log10(distance_mat)*regP.params['np.log10(distance_in_km)'])/regP.params['magnitude']
    median_mag_P = np.nanmedian(mag_estimate_P, axis=1)
    mean_mag_P = np.nanmean(mag_estimate_P, axis=1)
    std_mag_P = np.nanstd(mag_estimate_P, axis=1)

    mag_estimate_S = (np.log10(data_peak_mat_S+1e-12) - site_terms_S - np.log10(distance_mat)*regS.params['np.log10(distance_in_km)'])/regS.params['magnitude']
    median_mag_S = np.nanmedian(mag_estimate_S, axis=1)
    mean_mag_S = np.nanmean(mag_estimate_S, axis=1)
    std_mag_S = np.nanstd(mag_estimate_S, axis=1)

    mag_estimate_final = mag_estimate_P.copy()
    # mag_estimate_final[das_time_mat>=cvm_tt_ts] = mag_estimate_S[das_time_mat>=cvm_tt_ts]
    mag_estimate_final = np.nanmedian(np.array([mag_estimate_P, mag_estimate_S]), axis=0)

    return mag_estimate_P,mag_estimate_S, mag_estimate_final

mag_estimate_P, mag_estimate_S, mag_estimate_final = calculate_magnitude(regP, regS, site_terms_P, site_terms_S, data_peak_mat_P, data_peak_mat_S, distance_mat)
mag_estimate_P1, mag_estimate_S1, mag_estimate_final1 = calculate_magnitude(regP, regS, site_terms_P, site_terms_S, data_peak_mat_P, data_peak_mat_S, distance_mat1)
mag_estimate_P2, mag_estimate_S2, mag_estimate_final2 = calculate_magnitude(regP, regS, site_terms_P, site_terms_S, data_peak_mat_P, data_peak_mat_S, distance_mat2)

if show_loc == 1:
    mag_estimate_final = mag_estimate_final1
elif show_loc == 2:
    mag_estimate_final = mag_estimate_final2
elif show_loc == 3:
    mag_estimate_final = (mag_estimate_final1 + mag_estimate_final2)/2
else:
    mag_estimate_final = mag_estimate_final*np.nan

# combine both P and S to give final magnitude 
das_time_mat = das_time[:, np.newaxis]

median_mag = np.nanmedian(mag_estimate_final, axis=1)
mean_mag = np.nanmean(mag_estimate_final, axis=1)
std_mag = np.nanstd(mag_estimate_final, axis=1)

#%% 
# make real time plotting
time_rang_show = [0, 120]
if np.isnan(np.nanmin(tt_tp)):
    time_rang_show[0] = np.nanmin(tt_ts)-10
else:
    time_rang_show[0] = np.nanmin(tt_tp)-3
if not np.isnan(np.nanmax(tt_ts)):
    time_rang_show[1] = np.nanmedian(tt_ts)+10

cmap = plt.cm.get_cmap('OrRd', 6)

probability_mat_normalized = probability_mat/np.nansum(probability_mat, axis=1, keepdims=True)

time_step_plot=1
for i_time, time_pt in enumerate(das_time[::time_step_plot]):
    if time_pt < time_rang_show[0]:
        continue

    # the data to plot
    strain_rate_plot = strain_rate.copy()
    strain_rate_plot[das_time0>time_pt, :] = np.nan
    tt_tp_plot = tt_tp.flatten()
    tt_tp_plot[tt_tp_plot>time_pt] = np.nan
    tt_ts_plot = tt_ts.flatten()
    tt_ts_plot[tt_ts_plot>time_pt] = np.nan
    mag_estimate_final_plot = mag_estimate_final.copy()
    mag_estimate_final_plot[das_time>time_pt, :] = np.nan
    median_mag_plot = median_mag.copy()
    median_mag_plot[das_time>time_pt] = np.nan
    std_mag_plot = std_mag.copy()
    std_mag_plot[das_time>time_pt] = np.nan

    estimated_mag = np.nanmax(median_mag_plot)

    fig, ax = plt.subplots(2, 2, figsize=(18,12), gridspec_kw={'width_ratios': [2, 1]})
    ax = ax.flatten()

    # Strain
    pclip=99
    clipVal = np.percentile(np.absolute(strain_rate), pclip)
    gca = ax[0]
    clb = gca.imshow(strain_rate_plot.T, 
                extent=[das_time0[0], das_time0[-1], mag_estimate_P.shape[1], 0],
                aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'), interpolation='none')

    gca.plot(tt_tp_plot.flatten(), np.arange(len(tt_tp_plot)), '--k', linewidth=2, label='P')
    gca.plot(tt_ts_plot.flatten(), np.arange(len(tt_ts_plot)), '-k', linewidth=2, label='S')

    gca.set_xlabel('Time (s)')
    gca.set_ylabel('channel number')
    gca.set_xlim(time_rang_show[0], time_rang_show[1])
    gca.invert_yaxis()

    axins1 = inset_axes(gca,
                        width="2%",  # width = 50% of parent_bbox width
                        height="70%",  # height : 5%
                        loc='lower left')
    gca.set_title(f'time = {time_pt:.2f} s, estimated magnitude: {median_mag_plot[(i_time)*time_step_plot]:.2f}', fontsize=20)
    fig.colorbar(clb, cax=axins1, orientation="vertical", label='strain rate ($10^{-6}$/s)')


    gca = ax[2]
    clb = gca.imshow(mag_estimate_final_plot.T, 
                extent=[das_time[0], das_time[-1], mag_estimate_final.shape[1], 0],
                aspect='auto', vmin=0, vmax=6, cmap=cmap, interpolation='none')
    gca.plot(tt_tp_plot.flatten(), np.arange(len(tt_tp_plot)), '--k', linewidth=2)
    gca.plot(tt_ts_plot.flatten(), np.arange(len(tt_ts_plot)), '-k', linewidth=2)

    gca.set_xlabel('Time (s)')
    gca.set_ylabel('channel number')
    gca.set_xlim(time_rang_show[0], time_rang_show[1])
    gca.invert_yaxis()

    axins2 = inset_axes(gca,
                        width="2%",  # width = 50% of parent_bbox width
                        height="70%",  # height : 5%
                        loc='lower left')
    fig.colorbar(clb, cax=axins2, orientation="vertical", label='Magnitude')


    gca = ax[1]
    cbar = gca.imshow(np.reshape(probability_mat_normalized[i_time*time_step_plot, :], lat_grid.shape), aspect='auto', extent=[lon_grid.min(), lon_grid.max(), lat_grid.max(), lat_grid.min()])
    gca.plot(DAS_lon, DAS_lat, '-r')
    gca.plot(eq_lon, eq_lat, 'r*', markersize=18, zorder=10)
    
    if show_loc == 1:
        gca.plot(location_mat1[i_time*time_step_plot, 1], location_mat1[i_time*time_step_plot, 0], 'x', color='gold', markeredgewidth=3, markersize=18)
    if show_loc == 2:
        gca.plot(location_mat2[i_time*time_step_plot, 1], location_mat2[i_time*time_step_plot, 0], 'x', color='gold', markeredgewidth=3, markersize=18)
    if show_loc == 3:
        gca.plot(location_mat1[i_time*time_step_plot, 1], location_mat1[i_time*time_step_plot, 0], 'x', color='gold', markeredgewidth=3, markersize=18)
        gca.plot(location_mat2[i_time*time_step_plot, 1], location_mat2[i_time*time_step_plot, 0], 'x', color='gold', markeredgewidth=3, markersize=18)

    fig.colorbar(cbar, ax=gca, label='pdf')

    gca.set_xlabel('Lon')
    gca.set_ylabel('Lat')
    map_range = 0.8
    gca.set_xlim(eq_lon.mean()-map_range, eq_lon.mean()+map_range)
    gca.set_ylim(eq_lat.mean()-map_range, eq_lat.mean()+map_range)
    gca.set_title('Locating')


    gca=ax[3]
    if (time_pt <= np.nanmin(tt_ts)) and (time_pt >= np.nanmin(tt_tp)):
        gca.plot([np.nanmin(tt_tp), time_pt], [eq_mag.values*1.3, eq_mag.values*1.3], '--', color='gray', linewidth=5)
        gca.text(np.nanmin(tt_tp), eq_mag.values*1.4, 'Pilot estimation', fontsize=18)
    elif time_pt > np.nanmin(tt_ts):
        gca.plot([np.nanmin(tt_tp), np.nanmin(tt_ts)], [eq_mag.values*1.3, eq_mag.values*1.3], '--', color='gray', linewidth=5)
        gca.text(np.nanmin(tt_tp), eq_mag.values*1.4, 'Pilot estimation', fontsize=18)

    gca.plot(das_time, median_mag_plot, '-r', linewidth=4, alpha=1, zorder=3, label='mean')
    gca.plot(das_time, median_mag_plot-std_mag_plot, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
    gca.plot(das_time, median_mag_plot+std_mag_plot, '--r', linewidth=2, alpha=0.5, zorder=3)

    if not np.isnan(estimated_mag):
        gca.text(x=np.nanmax(tt_tp+2), y=estimated_mag+0.3, s=f'M {median_mag_plot[(i_time)*time_step_plot]:.2f}', color='r', fontsize=18)

    gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')

    # label the response time
    if time_pt >= DAS_response_time:
        gca.vlines(x=[DAS_response_time], ymax=10, ymin=0, label='DAS $t_R$', color='#36756E', linewidth=4)
    if time_pt >= SEIS_response_time:
        gca.vlines(x=[SEIS_response_time], ymax=10, ymin=0, label='Seismic $t_R$', color='#724819', linewidth=4)

    gca.set_yticks(np.arange(0, 9))
    gca.set_ylim(0, eq_mag.values*1.5)
    gca.set_xlim(time_rang_show[0], time_rang_show[1])
    gca.set_xlabel('Time (s)')
    gca.set_ylabel('Estimated Magnitude')
    gca.legend(loc=4)

    plt.subplots_adjust(hspace=0.2)

    # some final handling
    letter_list = [str(chr(k+97)) for k in range(0, 10)]
    k=0
    for ii in range(0, 4):
        ax[ii].annotate(f'({letter_list[k]})', xy=(-0.15, 1.0), xycoords=ax[ii].transAxes, fontsize=18)
        k+=1

    plt.savefig(fig_output_dir + f'/{region_label}_{test_event_id}_estimated_mag_image_{i_time:03n}.png')#, bbox_inches='tight')
    plt.close('all')

    if time_pt>time_rang_show[1]:
        break
# %%
# Combine into a movie
import imageio
fileList = []
for i_time, time_pt in enumerate(das_time[::time_step_plot]):
    if time_pt < time_rang_show[0]:
        continue
    complete_path = fig_output_dir + f'/{region_label}_{test_event_id}_estimated_mag_image_{i_time:03n}.png'
    fileList.append(complete_path)
    if time_pt>time_rang_show[1]:
        break
writer = imageio.get_writer(fig_output_dir + f'/{region_label}_{event_name}_movie.mp4', fps=1)

for im in fileList:
    writer.append_data(imageio.imread(im))
writer.close()
# %%
# Plot the last frame again as paper figure
fig, ax = plt.subplots(2, 2, figsize=(18,12), gridspec_kw={'width_ratios': [2, 1]})
ax = ax.flatten()

# Strain
pclip=99
clipVal = np.percentile(np.absolute(strain_rate), pclip)
gca = ax[0]
clb = gca.imshow(strain_rate_plot.T, 
            extent=[das_time0[0], das_time0[-1], mag_estimate_P.shape[1], 0],
            aspect='auto', vmin=-clipVal, vmax=clipVal, cmap=plt.get_cmap('seismic'), interpolation='none')

gca.plot(tt_tp_plot.flatten(), np.arange(len(tt_tp_plot)), '--k', linewidth=2, label='P')
gca.plot(tt_ts_plot.flatten(), np.arange(len(tt_ts_plot)), '-k', linewidth=2, label='S')

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.invert_yaxis()

axins1 = inset_axes(gca,
                    width="2%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='lower left')
gca.set_title(f'Final estimated magnitude: {median_mag_plot[(i_time)*time_step_plot]:.2f}', fontsize=20)
fig.colorbar(clb, cax=axins1, orientation="vertical", label='strain rate ($10^{-6}$/s)')


gca = ax[2]
clb = gca.imshow(mag_estimate_final_plot.T, 
            extent=[das_time[0], das_time[-1], mag_estimate_final.shape[1], 0],
            aspect='auto', vmin=0, vmax=6, cmap=cmap, interpolation='none')
gca.plot(tt_tp_plot.flatten(), np.arange(len(tt_tp_plot)), '--k', linewidth=2)
gca.plot(tt_ts_plot.flatten(), np.arange(len(tt_ts_plot)), '-k', linewidth=2)

gca.set_xlabel('Time (s)')
gca.set_ylabel('channel number')
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.invert_yaxis()

axins2 = inset_axes(gca,
                    width="2%",  # width = 50% of parent_bbox width
                    height="70%",  # height : 5%
                    loc='lower left')
fig.colorbar(clb, cax=axins2, orientation="vertical", label='Magnitude')


gca = ax[1]
cbar = gca.imshow(np.reshape(probability_mat_normalized[i_time*time_step_plot, :], lat_grid.shape), aspect='auto', extent=[lon_grid.min(), lon_grid.max(), lat_grid.max(), lat_grid.min()])
gca.plot(DAS_lon, DAS_lat, '-r')
gca.plot(eq_lon, eq_lat, 'r*', markersize=18, zorder=10)
if show_loc == 1:
    gca.plot(location_mat1[i_time*time_step_plot, 1], location_mat1[i_time*time_step_plot, 0], 'x', color='gold', markeredgewidth=3, markersize=18)
if show_loc == 2:
    gca.plot(location_mat2[i_time*time_step_plot, 1], location_mat2[i_time*time_step_plot, 0], 'x', color='gold', markeredgewidth=3, markersize=18)
if show_loc == 3:
    gca.plot(location_mat1[i_time*time_step_plot, 1], location_mat1[i_time*time_step_plot, 0], 'x', color='gold', markeredgewidth=3, markersize=18)
    gca.plot(location_mat2[i_time*time_step_plot, 1], location_mat2[i_time*time_step_plot, 0], 'x', color='gold', markeredgewidth=3, markersize=18)
fig.colorbar(cbar, ax=gca, label='pdf')

gca.set_xlabel('Lon')
gca.set_ylabel('Lat')
map_range = 0.8
gca.set_xlim(eq_lon.mean()-map_range, eq_lon.mean()+map_range)
gca.set_ylim(eq_lat.mean()-map_range, eq_lat.mean()+map_range)
gca.set_title('Locating')


gca=ax[3]
# gca.plot(das_time, mag_estimate_final, '-k', linewidth=0.1, alpha=0.1)
if (time_pt <= np.nanmin(tt_ts)) and (time_pt >= np.nanmin(tt_tp)):
    gca.plot([np.nanmin(tt_tp), time_pt], [eq_mag.values*1.3, eq_mag.values*1.3], '--', color='gray', linewidth=5)
    gca.text(np.nanmin(tt_tp), eq_mag.values*1.4, 'Pilot estimation', fontsize=18)
elif time_pt > np.nanmin(tt_ts):
    gca.plot([np.nanmin(tt_tp), np.nanmin(tt_ts)], [eq_mag.values*1.3, eq_mag.values*1.3], '--', color='gray', linewidth=5)
    gca.text(np.nanmin(tt_tp), eq_mag.values*1.4, 'Pilot estimation', fontsize=18)

gca.plot(das_time, median_mag_plot, '-r', linewidth=4, alpha=1, zorder=3, label='mean')
gca.plot(das_time, median_mag_plot-std_mag_plot, '--r', linewidth=2, alpha=0.5, zorder=3, label='STD')
gca.plot(das_time, median_mag_plot+std_mag_plot, '--r', linewidth=2, alpha=0.5, zorder=3)

if not np.isnan(estimated_mag):
    gca.text(x=np.nanmax(tt_tp+2), y=estimated_mag+0.3, s=f'M {median_mag_plot[(i_time)*time_step_plot]:.2f}', color='r', fontsize=18)

gca.hlines(y=[eq_mag], xmin=-10, xmax=120, color='green', label='catalog M')

# label the response time
gca.vlines(x=[DAS_response_time], ymax=10, ymin=0, label='DAS $t_R$', color='#36756E', linewidth=4)
gca.vlines(x=[SEIS_response_time], ymax=10, ymin=0, label='Seismic $t_R$', color='#724819', linewidth=4)

gca.set_yticks(np.arange(0, 9))
gca.set_ylim(0, eq_mag.values*1.5)
gca.set_xlim(time_rang_show[0], time_rang_show[1])
gca.set_xlabel('Time (s)')
gca.set_ylabel('Estimated Magnitude')
gca.legend(loc=4)

plt.subplots_adjust(hspace=0.2)

# some final handling
letter_list = [str(chr(k+97)) for k in range(0, 10)]
k=0
for ii in range(0, 4):
    ax[ii].annotate(f'({letter_list[k]})', xy=(-0.15, 1.0), xycoords=ax[ii].transAxes, fontsize=18)
    k+=1

plt.savefig(fig_output_dir + f'/{region_label}_{event_name}_estimated_mag_image_final.png')#, bbox_inches='tight')
plt.close('all')
# %%
