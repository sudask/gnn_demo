import xarray as xr  
import numpy as np

ds = xr.open_dataset("ncdata/Combined_TAIR_1.nc", engine="netcdf4") 

latitude = ds['LAT'].values
longitude = ds['LON'].values
tair = ds['TAIR'].values
timelist = ds['timelist'].values
interp_data = ds['interp_data'].values
targLatLon = ds['targLatLon'].values

lat = latitude[100:150]
lon = longitude[100:150]
real_vals = tair[:, 100:150, 100:150]

valid_indices = np.where((targLatLon[:, 0] <= lat[-1]) & (targLatLon[:, 1] <= lon[-1]) & (targLatLon[:, 0] >= lat[0]) & (targLatLon[:, 1] >= lon[0]))[0]
obs_station = targLatLon[valid_indices, :]
obs = interp_data[:, valid_indices]

def generateEdgeIndex(obs_station):
    edge_index = []
    for i in range(len(obs_station)):
        for j in range(len(obs_station)):
            if np.linalg.norm(obs_station[i] - obs_station[j]) < 1.5:
                edge_index.append([i, j])
    
    return np.array(edge_index).T

edge_index = generateEdgeIndex(obs_station)

np.save("data/edge_index.npy", edge_index)

np.save("data/lat.npy", lat)
np.save("data/lon.npy", lon)
np.save("data/vals.npy", real_vals)
np.save("data/station.npy", obs_station)
np.save("data/obs.npy", obs)