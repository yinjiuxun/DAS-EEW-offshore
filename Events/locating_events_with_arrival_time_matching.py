#%%
import pandas as pd
import numpy as np
import tqdm
import os

import sys
sys.path.append('../')
from utility.general import *
from utility.loading import load_phasenet_pick, load_event_data

from scipy.signal import find_peaks

# Plotting
import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 300,  # to adjust notebook inline plot size
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
# Define the path to store all the output results
output_dir = '../results/arrival_time_matching'
mkdir(output_dir)

#%%
# load the DAS channel location
das_info = pd.read_csv('../data_files/das_info/das_info.csv')
catalog = pd.read_csv('../data_files/catalogs/catalog.csv')

DAS_channel_num = das_info.shape[0]
DAS_index = das_info['index']
DAS_lon = das_info['longitude']
DAS_lat = das_info['latitude']

center_lon = np.mean(DAS_lon)
center_lat = np.mean(DAS_lat)

# %%
# prepare the "grid" point around the das array
n_xgrid, n_ygrid = 75, 75
num_points = n_xgrid * n_ygrid
x_min, x_max = center_lon-1.5, center_lon+1.5
y_min, y_max = center_lat-1.5, center_lat+1.5
xgrid_list = np.linspace(x_min, x_max, n_xgrid, endpoint=True)
ygrid_list = np.linspace(y_min, y_max, n_ygrid, endpoint=True)

lon_grid, lat_grid = np.meshgrid(xgrid_list, ygrid_list)

# %%
# show the location mesh
fig, ax = plt.subplots(figsize=(7, 6))
cmp = ax.scatter(DAS_lon, DAS_lat, s=10, c=DAS_index, cmap='jet')
ax.plot(center_lon, center_lat, 'r*')
ax.plot(lon_grid.flatten(), lat_grid.flatten(), 'k+')
ax.set_ylim(center_lat-1.1, center_lat+1.1)
ax.set_xlim(center_lon-1.1, center_lon+1.1)
fig.colorbar(cmp)

# %%
# Work out a handy travel time table to do interpolation
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees

travel_time_table_file =  '../data_files/travel_time_table/travel_time_table.npz'

# from one event to all channels
event_arrival_P = np.zeros((DAS_channel_num, num_points)) 
event_arrival_S = np.zeros((DAS_channel_num, num_points)) 
S_P_diff = np.zeros((DAS_channel_num, num_points)) 
distance_to_source_all = np.zeros((DAS_channel_num, num_points)) 

# First look for the precalculated TTT, if not exists, get one from interpolating TauP 
if not os.path.exists(travel_time_table_file):
    model = TauPyModel(model='iasp91')

    # distance list
    distance_fit = np.linspace(0, 2, 100)
    # depth list
    depth_fit = np.arange(0, 100, 1)

    distance_grid, depth_grid = np.meshgrid(distance_fit, depth_fit)

    tavel_time_P_grid = np.zeros(distance_grid.shape)
    tavel_time_S_grid = np.zeros(distance_grid.shape)

    for i_depth in tqdm(range(depth_grid.shape[0]), desc="Calculating arrival time..."):   

        for i_distance in range(distance_grid.shape[1]):
            try:
                arrivals = model.get_ray_paths(depth_fit[i_depth], distance_fit[i_distance], phase_list=['p', 's'])
                tavel_time_P_grid[i_depth, i_distance] = arrivals[0].time
                tavel_time_S_grid[i_depth, i_distance] = arrivals[1].time 
            except:
                tavel_time_P_grid[i_depth, i_distance] = np.nan
                tavel_time_S_grid[i_depth, i_distance] = np.nan

    # save the calculated Travel time table
    np.savez(travel_time_table_file, distance_grid=distance_grid, depth_grid=depth_grid, 
             tavel_time_p_grid=tavel_time_P_grid, tavel_time_s_grid=tavel_time_S_grid)

    print('Travel time table calculated!')
    
# The TTT calculated or already exists, directly load it.
temp = np.load(travel_time_table_file)
distance_grid = temp['distance_grid']
depth_grid = temp['depth_grid']
tavel_time_p_grid = temp['tavel_time_p_grid']
tavel_time_s_grid = temp['tavel_time_s_grid']


# save the calculated Travel time curves as templates
assumed_depth = 20
arrival_time_curve_template_file = f'../data_files/travel_time_table/arrival_time_template_{assumed_depth}km.npz'

if not os.path.exists(arrival_time_curve_template_file):
    # build the interpolation function
    from scipy.interpolate import interp2d, griddata

    ii = ~np.isnan(tavel_time_p_grid) # ignore the nan

    for i_eq in tqdm(range(num_points), desc="Calculating arrival time..."): 

            # estimate the arrival time of each earthquake to all channels
            P_arrival = np.zeros(DAS_channel_num)
            S_arrival = np.zeros(DAS_channel_num)
            distance_to_source = locations2degrees(DAS_lat, DAS_lon, lat_grid.flatten()[i_eq], lon_grid.flatten()[i_eq])

            P_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_p_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*assumed_depth))
            S_arrival = griddata(np.array([distance_grid[ii], depth_grid[ii]]).T, tavel_time_s_grid[ii], (distance_to_source, np.ones(distance_to_source.shape)*assumed_depth))

            distance_to_source_all[:, i_eq] = distance_to_source
            event_arrival_P[:, i_eq] = P_arrival
            event_arrival_S[:, i_eq] = S_arrival
            S_P_diff[:, i_eq] = S_arrival - P_arrival

    # save the calculated Travel time curves as templates
    np.savez(arrival_time_curve_template_file, lat_grid=lat_grid, lon_grid=lon_grid, distance_grid = distance_to_source_all,
                event_arrival_P=event_arrival_P, event_arrival_S=event_arrival_S, S_P_diff=S_P_diff)

#%%
# Load the calculated Travel time curves as templates
temp = np.load(arrival_time_curve_template_file)
distance_to_source_all = temp['distance_grid']
event_arrival_P_template = temp['event_arrival_P']
event_arrival_P_template_diff = np.diff(event_arrival_P_template, axis=0)
event_arrival_S_template = temp['event_arrival_S']
event_arrival_S_template_diff = np.diff(event_arrival_S_template, axis=0)
S_P_diff = event_arrival_S_template - event_arrival_P_template

# %%
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
# locating the events
event_id = 9006 # 9006, 9001
event_info = catalog[catalog.event_id == event_id]

tt_output_dir = '../data_files/pickings/picks_phasenet_das'
# Load the DAS data
strain_rate, info = load_event_data('../data_files/event_data', event_id)
das_dt = info['dt_s']
nt = strain_rate.shape[0]
das_time = np.arange(nt) * das_dt-30
time_range = None
pick_P, channel_P, pick_S, channel_S = load_phasenet_pick(tt_output_dir, event_id, das_time, das_info['index'], include_nan=True)

# remove some extreme outlies of PhaseNet-DAS picking
pick_P[abs(pick_P-np.nanmedian(pick_P))>2*np.nanstd(pick_P)] = np.nan
pick_S[abs(pick_S-np.nanmedian(pick_S))>2*np.nanstd(pick_S)] = np.nan

event_arrival_P_obs = pick_P
event_arrival_S_obs = pick_S
event_S_P_diff_obs = pick_S - pick_P

# only use the first 7000 channels that were well located
ii_channel = range(0, 7000)

# Fitting
_, norm_diff_P = match_arrival(event_arrival_P_obs[ii_channel], event_arrival_P_template[ii_channel, :], misfit_type='l1')
_, norm_diff_S = match_arrival(event_arrival_S_obs[ii_channel], event_arrival_S_template[ii_channel, :], misfit_type='l1')
_, norm_diff_SP = match_arrival(event_S_P_diff_obs[ii_channel], S_P_diff[ii_channel, :], misfit_type='l1', demean=False)

probability_P = misfit_to_probability(norm_diff_P)
probability_S = misfit_to_probability(norm_diff_S)
probability_SP = misfit_to_probability(norm_diff_SP)

#%%
# interference
probability_list = [probability_P*probability_SP]#, probability_SP, probability_P, probability_S]
phase_label_list = ['P+S-P']#, 'S-P', 'P', 'S']

for i in range(len(probability_list)):
    probability = probability_list[i]
    phase_label = phase_label_list[i]

    ii_max = np.nanargmax(probability)
    # peaks, _ = find_peaks(probability, distance=500, height=np.nanmax(probability)*0.6)
    # ii_max1 = peaks[0]
    # ii_max2 = peaks[1]

    # ii_max = ii_max2

    # plot the results
    fig, ax = plt.subplots(3, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
    ax[0].plot(DAS_lon, DAS_lat, '-b', label='DAS array', zorder=10)
    ax[0].plot(lon_grid.flatten()[ii_max], lat_grid.flatten()[ii_max], 'o', color='gold', label='optimal location', markersize=15, markeredgewidth=2, markerfacecolor='None')
    ax[0].plot(event_info.longitude, event_info.latitude, 'r*', markersize=20, label='catalog location')
    ax[0].set_ylim(center_lat-1.5, center_lat+1.5)
    ax[0].set_xlim(center_lon-1.5, center_lon+1.5)

    cbar = ax[0].imshow(np.reshape(probability, (75,75)), aspect='auto', extent=[lon_grid.min(), lon_grid.max(), lat_grid.max(), lat_grid.min()])
    ax[0].set_title(f'Earthquake location, assumed depth at {assumed_depth} km')
    ax[0].legend(fontsize=10, loc=2)
    fig.colorbar(cbar, ax=ax[0], label='pdf')

    ax[1].plot(event_arrival_P_obs[ii_channel] - np.nanmean(event_arrival_P_obs[ii_channel]), '.r', label='observed P', markersize=3)
    ax[1].plot(event_arrival_P_template[ii_channel, ii_max] - np.nanmean(event_arrival_P_template[ii_channel, ii_max]), '-r', label='Matched P')
    ax[1].plot(event_arrival_S_obs[ii_channel] - np.nanmean(event_arrival_S_obs[ii_channel]), '.b', label='observed S', markersize=3)
    ax[1].plot(event_arrival_S_template[ii_channel, ii_max] - np.nanmean(event_arrival_S_template[ii_channel, ii_max]), '-b', label='Matched S')
    ax[1].set_xlabel('Channels')
    ax[1].set_ylabel('Demeaned \narrival time (s)')
    ax[1].legend(fontsize=10)
    origin_time_diff = np.nanmedian(event_arrival_P_obs[ii_channel] - event_arrival_P_template[ii_channel, ii_max])
    if phase_label == 'P+S-P':
        ax[1].set_title(f'Origin time error: {origin_time_diff:.2f} s')

    ax[2].plot(event_S_P_diff_obs, '.k', label='observed tS - tP', markersize=3)
    ax[2].plot(S_P_diff[:, ii_max], '-k', label='Matched tS - tP')
    ax[2].set_xlabel('Channels')
    ax[2].set_ylabel('Arrival time \ndifference (s)')
    ax[2].legend(fontsize=10)

    plt.savefig(output_dir + f'/event_{event_id}_arrival_matching_{phase_label}_{assumed_depth}km.png', bbox_inches='tight')

# %%
# Try Wadati diagram
P_X0 = event_arrival_P_template[:, ii_max]#event_arrival_P_obs 
S_P0 = event_arrival_S_template[:, ii_max] - event_arrival_P_template[:, ii_max]#event_arrival_S_obs - event_arrival_P_obs

P_X = event_arrival_P_obs[ii_channel] - origin_time_diff
S_P = event_arrival_S_obs[ii_channel] - event_arrival_P_obs[ii_channel]

ii = np.isnan(P_X)
jj = np.isnan(S_P)
pp = abs(P_X - np.nanmean(P_X))<=1.5*np.nanstd(P_X)
qq = abs(S_P - np.nanmean(S_P))<=1.5*np.nanstd(S_P)
kk = ~((ii) | (jj)) & pp & qq

# Use numpy to fit a linear relation
coefficients0 = np.polyfit(P_X[kk], S_P[kk], 1)
plt.plot(P_X0, S_P0, 'r-', label='matched')
plt.plot(P_X[kk], S_P[kk], 'b.', alpha=0.01)
plt.plot(np.nan, np.nan, 'b.', label='picked')
plt.plot(P_X[kk], P_X[kk]*coefficients0[0]+coefficients0[1], 'b-', label=f'y={coefficients0[0]:.2f}x + {coefficients0[1]:.2f}')
plt.ylim(2, 6)
plt.xlabel('tP (s)')
plt.ylabel('tS - tP (s)')
plt.text(P_X[kk].max(), S_P[kk].max(), s=f'Vp/Vs = {coefficients0[0]+1:.2f}')
# plt.text(5, 5.5, s=f'Origin time error = {-coefficients0[1]/coefficients0[0]:.2f}s')
plt.legend(loc=2, fontsize=12)
if event_id == 9001:
    plt.title('La Ligua earthquake')
elif event_id == 9006:
    plt.title('Vaparaíso earthquake')

plt.savefig(output_dir + f'/event_{event_id}_Wadati_diagram.png', bbox_inches='tight')

# %%
