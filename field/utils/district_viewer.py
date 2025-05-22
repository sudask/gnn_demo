import numpy as np
import matplotlib.pyplot as plt

LAT_SIZE = 36
LON_SIZE = 50
MIN_LAT_INDEX = 142
MIN_LON_INDEX = 311
TIME_INDEX = 5

def plot(coordinate, real_vals, obs_pos, obs_vals):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coordinate[:, 0], coordinate[:, 1], real_vals, c='blue', label='Real Values', alpha=0.1)
    ax.scatter(obs_pos[:, 0], obs_pos[:, 1], obs_vals, marker='s', c='g', label='obs station', alpha=0.9)

    ax.set_title('Real and Predict Values in 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()

all_lat = np.load("data/lat.npy").astype(np.float32) # 320
all_lon = np.load("data/lon.npy").astype(np.float32) # 416
all_station = np.load("data/station_pos.npy").astype(np.float32) # 1754 * 2
# from here on, depends on time
all_val = np.load("data/vals.npy").astype(np.float32) # 1995 * 320 * 416
all_obs = np.load("data/obs.npy").astype(np.float32) # 1995 * 1754
all_time = np.load("data/time.npy") # 1995 * 1

lat = all_lat[MIN_LAT_INDEX:MIN_LAT_INDEX+LAT_SIZE]
lon = all_lon[MIN_LON_INDEX:MIN_LON_INDEX+LON_SIZE]

x, y = np.meshgrid(lat, lon, indexing='ij')
coordinate = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

real_vals = all_val[TIME_INDEX, MIN_LAT_INDEX:MIN_LAT_INDEX+LAT_SIZE, MIN_LON_INDEX:MIN_LON_INDEX+LON_SIZE].reshape(-1)
# choose obs stations belong to required range
valid_indices = np.where((all_station[:, 0] <= lat[-1]) & (all_station[:, 1] <= lon[-1]) & (all_station[:, 0] >= lat[0]) & (all_station[:, 1] >= lon[0]))[0]
obs_station = all_station[valid_indices, :]
obs = all_obs[TIME_INDEX, valid_indices]

print(f"lat size: {LAT_SIZE} | lon size: {LON_SIZE} | Amout of obs station: {obs_station.shape[0]} | lat range: {lat[0]} - {lat[-1]} | lon range: {lon[0]} - {lon[-1]} | Time: {all_time[TIME_INDEX]}")

plot(coordinate, real_vals, obs_station, obs)
