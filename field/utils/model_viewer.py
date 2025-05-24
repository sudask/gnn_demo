import json
import torch
from mymodel import GeneralModel
import numpy as np

config_file = "./config/config_36_50.json"

with open(config_file) as f:
    config = json.load(f)

LAT_SIZE = config["lat_size"]
LON_SIZE = config["lon_size"]
MIN_LAT_INDEX = config["min_lat_index"]
MIN_LON_INDEX = config["min_lon_index"]
NUM_DATA = config["num_data"]
STEP_SIZE = config["step_size"]
GAMMA = config["gamma"]
NUM_EPOCH = config["num_epoch"]

# ======================== generate indices of different sets ========================

# split ratio
train_ratio = 0.6
validate_ratio = 0.2
test_ratio = 0.2

# Calculate sizes
train_size = int(NUM_DATA * train_ratio)
validate_size = int(NUM_DATA * validate_ratio)
test_size = NUM_DATA - train_size - validate_size

numbers = list(range(NUM_DATA))

training_indices = random.sample(numbers, train_size)
remaining_indices = [i for i in numbers if i not in set(training_indices)]
validation_indices = random.sample(remaining_indices, validate_size)
testing_indices = [i for i in remaining_indices if i not in set(validation_indices)]

# ======================== load data from npy files ========================

all_lat = np.load("data/lat.npy").astype(np.float32) # 320
all_lon = np.load("data/lon.npy").astype(np.float32) # 416
all_station = np.load("data/station_pos.npy").astype(np.float32) # 1754 * 2
# from here on, depends on time
all_val = np.load("data/vals.npy").astype(np.float32) # 1995 * 320 * 416
all_obs = np.load("data/obs.npy").astype(np.float32) # 1995 * 1754
all_time = np.load("data/time.npy").astype(np.float32) # 1995 * 1

# ======================== use indices to genearte data sets ========================

lat = all_lat[MIN_LAT_INDEX:MIN_LAT_INDEX+LAT_SIZE]
lon = all_lon[MIN_LON_INDEX:MIN_LON_INDEX+LON_SIZE]

real_vals = all_val[training_indices, MIN_LAT_INDEX:MIN_LAT_INDEX+LAT_SIZE, MIN_LON_INDEX:MIN_LON_INDEX+LON_SIZE]
# choose obs stations belong to required range
valid_indices = np.where((all_station[:, 0] <= lat[-1]) & (all_station[:, 1] <= lon[-1]) & (all_station[:, 0] >= lat[0]) & (all_station[:, 1] >= lon[0]))[0]
obs_station = all_station[valid_indices, :]
obs = all_obs[training_indices][:, valid_indices]

print("Data info: ")

processed_data = []
for i in range(NUM_DATA):
    obs_reshaped = all_obs[i, valid_indices].reshape(-1, 1)
    feature = torch.from_numpy(np.concatenate((obs_reshaped, obs_station), axis=1))
    vals = torch.from_numpy(all_val[i, MIN_LAT_INDEX:MIN_LAT_INDEX+LAT_SIZE, MIN_LON_INDEX:MIN_LON_INDEX+LON_SIZE].reshape(-1))
    processed_data.append(MyData(feature, torch.from_numpy(edge_index), vals))


training_data = [processed_data[i] for i in training_indices]
validation_data = [processed_data[i] for i in validation_indices]
testing_data = [processed_data[i] for i in testing_indices]
