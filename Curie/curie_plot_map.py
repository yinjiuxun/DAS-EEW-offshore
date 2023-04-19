#%% import modules
import pandas as pd
import sys
sys.path.append('../')
from utility.general import mkdir
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

import pygmt
#%% 
# make the output directory
figure_output_dir = '../results'
mkdir(figure_output_dir)
#%% 
# load DAS info
DAS_info = pd.read_csv('../data_files/das_info/das_info.csv')
DAS_channel_num = DAS_info.shape[0]
DAS_lon = DAS_info.longitude
DAS_lat = DAS_info.latitude

# Load catalog
catalog = pd.read_csv('../data_files/catalogs/catalog.csv')
event_id_selected = [9001, 9007, 9006]
catalog_select_all = catalog[catalog.event_id.isin(event_id_selected)]

num_events = catalog_select_all.shape[0]
event_lon = catalog_select_all.longitude
event_lat = catalog_select_all.latitude
event_id = catalog_select_all.event_id

# load the regional permanent station
stations = pd.read_csv('../data_files/nearby_stations/curie_nearby_stations.csv', index_col=None)
moment_tensor_catalog = pd.read_csv('../data_files/moment_tensor_catalog/moment_tensor_catalog.csv', index_col=None, header=None)

plt.hist(moment_tensor_catalog[2])
plt.xlabel('depth (km)')
plt.ylabel('Counts')
plt.title(f'mean: {moment_tensor_catalog[2].mean():.2f}km, median: {moment_tensor_catalog[2].median():.2f}km')

#%%
# =========================  Plot both arrays in Chile with PyGMT ==============================
plt.close('all')
gmt_region = [-72.5, -70.9, -33.3, -31.8]

projection = "M12c"
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=gmt_region)

# calculate the reflection of a light source projecting from west to east
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30])

fig = pygmt.Figure()
# define figure configuration
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT_ANNOT_PRIMARY=12, FONT_TITLE="14p,Helvetica,black")

# --------------- plotting the original Data Elevation Model -----------
fig.basemap(region=gmt_region, 
projection=projection, 
frame=['WSne', "x0.5", "y0.5"]
)
pygmt.makecpt(cmap="geo", series=[-4000, 4000])
fig.grdimage(
    grid=grid,
    projection=projection,
    cmap=True,
    shading='+a45+nt1',
    transparency=35
)

fig.plot(x=stations.Longitude.astype('float'), y=stations.Latitude.astype('float'), style="i0.8c", color="darkred")
fig.plot(x=catalog_select_all.longitude.astype('float'), y=catalog_select_all.latitude.astype('float'), style="c0.3c", color="black")
for ii in range(catalog_select_all.shape[0]):
    fig.text(text=catalog_select_all.iloc[ii, :].place, x=catalog_select_all.iloc[ii, :].longitude.astype('float'), 
        y=catalog_select_all.iloc[ii, :].latitude.astype('float')-0.05, font="10p,Helvetica-Bold,black")
fig.plot(x=DAS_info.longitude[::100].astype('float'), y=DAS_info.latitude[::100].astype('float'), style="c0.05c", color="red")
fig.text(text="DAS array", x=-71.55, y=-32.9, font="12p,Helvetica-Bold,red")
fig.text(text="C1.VA01", x=-71.5, y=-33.05, font="12p,Helvetica-Bold,black")

fig.show()
fig.savefig(figure_output_dir + '/map_of_earthquakes_Curie_GMT_0.png')
fig.savefig(figure_output_dir + '/map_of_earthquakes_Curie_GMT_0.pdf')
