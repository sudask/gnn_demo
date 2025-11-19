import json
from config import*
from mymodel import*
from for_plot import*
from for_data import*
from train import*
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR

# ======================== import config info from json ========================
config_file = "./config/config_36_50.json"

with open(config_file) as f:
    config = json.load(f)

random.seed(config["random_seed"])
LAT_SIZE = config["lat_size"]
LON_SIZE = config["lon_size"]
MIN_LAT_INDEX = config["min_lat_index"]
MIN_LON_INDEX = config["min_lon_index"]
NUM_DATA = config["num_data"]
STEP_SIZE = config["step_size"]
GAMMA = config["gamma"]
NUM_EPOCH = config["num_epoch"]

# ======================== load data from npy files ========================
all_lat = np.load("data/lat.npy").astype(np.float32) # 320
all_lon = np.load("data/lon.npy").astype(np.float32) # 416
all_station = np.load("data/station_pos.npy").astype(np.float32) # 1754 * 2
# from here on, depends on time
all_val = np.load("data/vals.npy").astype(np.float32) # 1995 * 320 * 416
all_obs = np.load("data/obs.npy").astype(np.float32) # 1995 * 1754
all_time = np.load("data/time.npy").astype(np.float32) # 1995 * 1

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

# ======================== use indices to genearte data sets ========================
u_lat = all_lat[MIN_LAT_INDEX:MIN_LAT_INDEX+LAT_SIZE]
u_lon = all_lon[MIN_LON_INDEX:MIN_LON_INDEX+LON_SIZE]

# choose obs stations belong to required range
valid_indices = np.where((all_station[:, 0] <= u_lat[-1]) & (all_station[:, 1] <= u_lon[-1]) & (all_station[:, 0] >= u_lat[0]) & (all_station[:, 1] >= u_lon[0]))[0]
u_obs_station = all_station[valid_indices, :]
obs = all_obs[training_indices][:, valid_indices]

print("Data info: ")
print(f"training size: {train_size} | validation size: {validate_size} | testing size: {test_size} | obs amount: {obs.shape[1]}")

edge_index = generateEdgeIndex(u_obs_station)
# plotObs(u_lat, u_lon, u_obs_station, edge_index)
# exit()

# noramlize
training_temperature = all_val[training_indices, MIN_LAT_INDEX:MIN_LAT_INDEX+LAT_SIZE, MIN_LON_INDEX:MIN_LON_INDEX+LON_SIZE]
min_temp = np.min(training_temperature)
max_temp = np.max(training_temperature)
min_lat = all_lat[MIN_LAT_INDEX]
max_lat = all_lat[MIN_LAT_INDEX+LAT_SIZE-1]
min_lon = all_lon[MIN_LON_INDEX]
max_lon = all_lon[MIN_LON_INDEX+LON_SIZE-1]

u_obs = all_obs[:NUM_DATA][:, valid_indices]
u_val = all_val[:NUM_DATA, MIN_LAT_INDEX:MIN_LAT_INDEX+LAT_SIZE, MIN_LON_INDEX:MIN_LON_INDEX+LON_SIZE]
norm_obs = (u_obs - min_temp) / (max_temp - min_temp)
norm_lat = (u_lat - min_lat)  / (max_lat - min_lat)
norm_lon = (u_lon - min_lon)  / (max_lon - min_lon)
norm_val = (u_val - min_temp) / (max_temp - min_temp)

norm_sta = u_obs_station
norm_sta[:, 0] -= min_lat
norm_sta[:, 0] /= (max_lat - min_lat)
norm_sta[:, 1] -= min_lon
norm_sta[:, 1] /= (max_lon - min_lon)

# assemble data into MyData
processed_data = []
for i in range(NUM_DATA):
    obs_reshaped = norm_obs[i, :].reshape(-1, 1)
    feature = torch.from_numpy(np.concatenate((obs_reshaped, norm_sta), axis=1))
    vals = torch.from_numpy(norm_val[i].reshape(-1))
    processed_data.append(MyData(feature, torch.from_numpy(edge_index), vals))


training_data = [processed_data[i] for i in training_indices]
validation_data = [processed_data[i] for i in validation_indices]
testing_data = [processed_data[i] for i in testing_indices]

# ======================== model ========================

model = GeneralModel(obs.shape[1], LAT_SIZE * LON_SIZE)

# ======================== set nessesary components ========================

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# ======================== different schedulers ========================

scheduler1 = ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.8,
    patience=5
)

scheduler2 = CyclicLR(
    optimizer,
    base_lr=0.01,
    max_lr=0.1,
    step_size_up=100,
    mode='triangular'
)

scheduler3 = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ======================== traing and svae model ========================

save_path = f"checkpoints/model_{LAT_SIZE}_{LON_SIZE}.pth"
loss_history = train(model, training_data, validation_data, optimizer, scheduler3, criterion, NUM_EPOCH, save_path)
# plotLossCurve(loss_history)

# ======================== display results ========================

checkpoint = torch.load(save_path, weights_only=True)
model.load_state_dict(checkpoint)

x, y = np.meshgrid(u_lat, u_lon, indexing='ij')
coordinate = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

mse_error = np.zeros(LAT_SIZE * LON_SIZE)
for data in testing_data:
    predict = model(data)
    predict_val = predict.detach().numpy()
    real_val = data.vals.detach().numpy()
    
    # denormalize
    denormalized_predict_val = predict_val * (max_temp - min_temp) + min_temp
    denormalized_real_val = real_val * (max_temp - min_temp) + min_temp
    diff = denormalized_real_val - denormalized_predict_val
    diff2 = diff ** 2
    mse_error += diff2

mse = mse_error / len(testing_data)
print("Average mse: ", np.mean(mse_error))

# plotError(coordinate, mse)

idx = 300
real_val = testing_data[idx].vals.detach().numpy()
predict_val = model(testing_data[idx]).detach().numpy()
feature = testing_data[idx].feature.detach().numpy()

# denormalize before plot
denormalized_predict_val = predict_val * (max_temp - min_temp) + min_temp
denormalized_real_val = real_val * (max_temp - min_temp) + min_temp
obs_info = np.zeros_like(feature)
obs_info[:, 0] = feature[:, 0] * (max_temp - min_temp) + min_temp
obs_info[:, 1] = feature[:, 1] * (max_lat  - min_lat) + min_lat
obs_info[:, 2] = feature[:, 2] * (max_lon  - min_lon) + min_lon

plot3d(coordinate, real_val, predict_val, obs_info)
# plot_compare_3d(coordinate, real_val, predict_val)

