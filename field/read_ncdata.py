import xarray as xr  
import numpy as np

ds1 = xr.open_dataset("ncdata/Combined_TAIR_1.nc", engine="netcdf4")

latitude = ds1['LAT'].values # 320
longitude = ds1['LON'].values # 416
targLatLon = ds1['targLatLon'].values # 1754 * 2, obs stations' pos

res_tair = ds1['TAIR'].values
res_time = ds1['timelist'].values.reshape(-1, 1)
res_obs = ds1['interp_data'].values
print(res_time.shape)
for i in range(2, 11):
    ds = xr.open_dataset(f"ncdata/Combined_TAIR_{i}.nc", engine="netcdf4")

    tair = ds['TAIR'].values # 200 * 320 * 416, val at each time each pos
    timelist = ds['timelist'].values.reshape(-1, 1) # 200
    interp_data = ds['interp_data'].values # 200 * 1754, val on 1754 obs stations at each time

    res_tair = np.vstack((res_tair, tair))
    res_time = np.vstack((res_time, timelist))
    res_obs = np.vstack((res_obs, interp_data))

np.save("data/lat.npy", latitude)
np.save("data/lon.npy", longitude)
np.save("data/station_pos", targLatLon)
np.save(f"data/vals.npy", res_tair)
np.save(f"data/time.npy", res_time)
np.save(f"data/obs.npy", res_obs)