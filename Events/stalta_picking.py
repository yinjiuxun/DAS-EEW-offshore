#%% import modules

import pandas as pd
#from sep_util import read_file
import numpy as np
import sys 
import tqdm

import sys
sys.path.append('../')
from utility.general import mkdir
from utility.loading import load_event_data

# Plotting
import matplotlib
import matplotlib.pyplot as plt

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


event_folder = '../data_files/event_data'  
vm_tt_dir = '../data_files/pickings/theoretical_arrival_time0'
mv_tt_dir = '../data_files/pickings/picks_phasenet_das'
catalog = pd.read_csv('../data_files/catalogs/catalog.csv')
das_info = pd.read_csv('../data_files/das_info/das_info.csv')
fig_folder = '../results/STALTA_pickings'
mkdir(fig_folder)

# %%
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, carl_sta_trig, coincidence_trigger, plot_trigger

event_id = 9007
data, info = load_event_data(event_folder, event_id)
DAS_channel_num = data.shape[1]
dt = info['dt_s']
das_time = np.arange(data.shape[0])*dt-30

# load tt from velocity model
vm_tt_1d = pd.read_csv(vm_tt_dir + f'/1D_tt_{event_id}.csv')
# load pickings from phasenet-das
ml_picking = pd.read_csv(mv_tt_dir + f'/{event_id}.csv')
ml_picking = ml_picking[ml_picking.channel_index < DAS_channel_num]
ml_picking_P = ml_picking[ml_picking.phase_type == 'P']
ml_picking_S = ml_picking[ml_picking.phase_type == 'S']
ml_tt_tp = np.zeros(shape=DAS_channel_num)*np.nan
ml_tt_ts = ml_tt_tp.copy()
ml_tt_tp[ml_picking_P.channel_index] = das_time[ml_picking_P.phase_index]
ml_tt_ts[ml_picking_S.channel_index] = das_time[ml_picking_S.phase_index]

stalta_ratio = np.zeros(data.shape)

# Parameters of the STA/LTA
f_sample = 1/dt

threshold_coincidence = 1
short_term = 0.1
long_term = 2
trigger_on = 6
trigger_off = 2

npt_extend = 500
for i in tqdm.tqdm(range(data.shape[1])):
    test_data = data[:,i]
    stalta_ratio_temp = recursive_sta_lta(test_data, int(short_term * f_sample), int(long_term * f_sample))
    #stalta_ratio_temp = classic_sta_lta(test_data, int(short_term * f_sample), int(long_term * f_sample))
    # stalta_ratio[:, i] = stalta_ratio_temp[(npt_extend):(-npt_extend)]
    stalta_ratio[:, i] = stalta_ratio_temp

# cft = classic_sta_lta(st[0].data, int(short_term * f_sample), int(long_term * f_sample))
# cft = carl_sta_trig(st[0].data, int(short_term * 10), int(long_term * 10), 0.8, 0.8)
#%%
threshold = '_thresholded'
thresholod = 6

stalta_ratio_threshold = stalta_ratio.copy()
if threshold == '_thresholded':
    stalta_ratio_threshold[stalta_ratio_threshold<=thresholod]=0

stalta_ratio_peaks = np.max(stalta_ratio, axis=0)
ii_wrong_peak = stalta_ratio_peaks<=thresholod
stalta_ratio_peaks_index = np.argmax(stalta_ratio, axis=0)
peak_time = das_time[stalta_ratio_peaks_index[np.newaxis, :]]
peak_time[:, ii_wrong_peak] = np.nan

fig, ax = plt.subplots(figsize=(12, 8))
clb = ax.imshow(stalta_ratio_threshold, extent=[0, stalta_ratio.shape[1], das_time[-1], das_time[0]],aspect='auto', vmin=0, vmax=thresholod)
ax.plot(vm_tt_1d.P_arrival, '--g', linewidth=2)
ax.plot(vm_tt_1d.S_arrival, '-g', linewidth=2, label='velocity model predicted')
ax.plot(ml_tt_tp, '--y', linewidth=2)
ax.plot(ml_tt_ts, '-y', linewidth=2, label='PhaseNet-DAS picking')
ax.legend(loc=2)

ax.set_ylim(vm_tt_1d.P_arrival.min()-5, vm_tt_1d.S_arrival.max()+5)
ax.set_ylabel('Time (s)')
ax.set_xlabel('Channel number')

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(clb, cax=cbar_ax,  orientation="vertical", label='STA/LTA')
plt.savefig(fig_folder + f'/{event_id}_STALTA_picking{threshold}.png', bbox_inches='tight')
# %%
