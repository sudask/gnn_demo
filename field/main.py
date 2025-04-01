from config import*
from mymodel import*
from plot import*
from train import*
from read_and_save_ncdata import*

BLOCK_SIZE = 80 # differenct size should use different model
NUM_TRAINING_DATA = 1500

# random sample
random.seed(42)
numbers = list(range(1800))
training_indices = random.sample(numbers, NUM_TRAINING_DATA)
testing_indices = [i for i in numbers if i not in set(training_indices)]

# load data
all_lat = np.load("data/lat.npy").astype(np.float32) # 320
all_lon = np.load("data/lon.npy").astype(np.float32) # 416
all_station = np.load("data/station_pos.npy").astype(np.float32) # 1754 * 2
# from here on, depends on time
all_val = np.load("data/vals.npy").astype(np.float32) # 1995 * 320 * 416
all_obs = np.load("data/obs.npy").astype(np.float32) # 1995 * 1754
all_time = np.load("data/time.npy").astype(np.float32) # 1995 * 1

# prepare data
lat = all_lat[30:30 + BLOCK_SIZE]
lon = all_lon[120:120 + BLOCK_SIZE]
# use 1500 time
real_vals = all_val[training_indices, 30:30 + BLOCK_SIZE, 120:120 + BLOCK_SIZE]
# choose obs stations belong to required range
valid_indices = np.where((all_station[:, 0] <= lat[-1]) & (all_station[:, 1] <= lon[-1]) & (all_station[:, 0] >= lat[0]) & (all_station[:, 1] >= lon[0]))[0]
obs_station = all_station[valid_indices, :]
obs = all_obs[training_indices][:, valid_indices]
print(f"Amount of obs stations: {obs.shape[1]}")

edge_index = generateEdgeIndex(obs_station)
# plotObs(lat, lon, obs_station)

training_data = []
for i in range(NUM_TRAINING_DATA):
    feature = torch.from_numpy(obs[i]).reshape(-1, 1)
    vals = torch.from_numpy(real_vals[i].reshape(-1))
    training_data.append(MyData(feature, torch.from_numpy(edge_index), vals))

model = None
if (obs.shape[1] == 94):
    model = model94()

if (obs.shape[1] == 415):
    model = model415()

if (obs.shape[1] == 19):
    model = model19()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

trainAndPlot(model, training_data, optimizer, criterion)

checkpoint = torch.load("checkpoints/test.pth", weights_only=True)
model.load_state_dict(checkpoint)

total_error = np.zeros(BLOCK_SIZE ** 2)
for i in testing_indices:
    feature = torch.from_numpy(all_obs[i, valid_indices]).reshape(-1, 1)
    testing_data = MyData(feature, torch.from_numpy(edge_index))

    predict = model(testing_data)
    predict_val = predict.detach().numpy()
    real_val = all_val[i, 30:30 + BLOCK_SIZE, 120:120 + BLOCK_SIZE].reshape(-1)

    error = (real_val - predict_val) ** 2
    total_error += error

avg_error = total_error / 300
print(avg_error.shape)

x, y = np.meshgrid(lat, lon, indexing='ij')
coordinate = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

plotError(coordinate, avg_error)
# plot3d(coordinate, real_val, predict_val)
# plot_compare_3d(coordinate, real_val, predict_val)

