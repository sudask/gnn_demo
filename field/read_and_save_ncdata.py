import xarray as xr  
import numpy as np

def generateEdgeIndex(obs_station):
    edge_index = []
    for i in range(len(obs_station)):
        for j in range(len(obs_station)):
            if np.linalg.norm(obs_station[i] - obs_station[j]) < 1.5:
                edge_index.append([i, j])
    
    return np.array(edge_index).T

def generateEdgeIndexAndWeight(obs_station):
    edge_index = []
    edge_weight = []
    for i in range(len(obs_station)):
        for j in range(len(obs_station)):
            distance = np.linalg.norm(obs_station[i] - obs_station[j])
            if distance < 1.5:
                edge_index.append([i, j])
                edge_weight.append(1.0 / (distance + 1e-5))  # 避免除以0
                
    edge_index = np.array(edge_index).T
    edge_weight = np.array(edge_weight)
    return edge_index, edge_weight

def prepareNcdata(block_size = 50):
    ds = xr.open_dataset("ncdata/Combined_TAIR_1.nc", engine="netcdf4")
    latitude = ds['LAT'].values # 320
    longitude = ds['LON'].values # 416
    tair = ds['TAIR'].values # 200 * 320 * 416, val at each time each pos
    timelist = ds['timelist'].values # 200
    interp_data = ds['interp_data'].values # 200 * 1754, val on 1754 obs stations at each time
    targLatLon = ds['targLatLon'].values # 1754 * 2, obs stations' pos

    print(f"Amount of latitudes and longitudes: {block_size}")
    lat = latitude[30:30 + block_size]
    lon = longitude[120:120 + block_size]
    real_vals = tair[:, 30:30 + block_size, 120:120 + block_size]

    valid_indices = np.where((targLatLon[:, 0] <= lat[-1]) & (targLatLon[:, 1] <= lon[-1]) & (targLatLon[:, 0] >= lat[0]) & (targLatLon[:, 1] >= lon[0]))[0]
    obs_station = targLatLon[valid_indices, :]
    obs = interp_data[:, valid_indices]
    print(f"Amount of obs stations: {obs.shape[1]}")

    edge_index = generateEdgeIndex(obs_station)

    np.save(f"data/edge_index_{block_size}.npy", edge_index)

    np.save(f"data/lat_{block_size}.npy", lat)
    np.save(f"data/lon_{block_size}.npy", lon)
    np.save(f"data/vals_{block_size}.npy", real_vals)
    np.save(f"data/station_{block_size}.npy", obs_station)
    np.save(f"data/obs_{block_size}.npy", obs)