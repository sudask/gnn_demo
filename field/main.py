from config import*
from mymodel import*
from plot import*
from train import*
from read_and_save_ncdata import*
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR

# ======================== generate indices of different sets ========================

random.seed(42)

BLOCK_SIZE = 80 # different size should use different model
NUM_DATA = 200

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

lat = all_lat[30:30 + BLOCK_SIZE]
lon = all_lon[120:120 + BLOCK_SIZE]

real_vals = all_val[training_indices, 30:30 + BLOCK_SIZE, 120:120 + BLOCK_SIZE]
# choose obs stations belong to required range
valid_indices = np.where((all_station[:, 0] <= lat[-1]) & (all_station[:, 1] <= lon[-1]) & (all_station[:, 0] >= lat[0]) & (all_station[:, 1] >= lon[0]))[0]
obs_station = all_station[valid_indices, :]
obs = all_obs[training_indices][:, valid_indices]

print("Data info: ")
print(f"training size: {train_size} | validation size: {validate_size} | testing size: {test_size} | obs amount: {obs.shape[1]}")

edge_index = generateEdgeIndex(obs_station)
# plotObs(lat, lon, obs_station, edge_index)

# exit()

training_data = []
validation_data = []
testing_data = []

for i in training_indices:
    obs_reshaped = all_obs[i, valid_indices].reshape(-1, 1)
    feature = torch.from_numpy(np.concatenate((obs_reshaped, obs_station), axis=1))
    # no lat and lon info:
    # feature = torch.from_numpy(obs[i]).reshape(-1, 1)
    vals = torch.from_numpy(all_val[i, 30:30 + BLOCK_SIZE, 120:120 + BLOCK_SIZE].reshape(-1))
    training_data.append(MyData(feature, torch.from_numpy(edge_index), vals))

for i in validation_indices:
    obs_reshaped = all_obs[i, valid_indices].reshape(-1, 1)
    feature = torch.from_numpy(np.concatenate((obs_reshaped, obs_station), axis=1))
    vals = torch.from_numpy(all_val[i, 30:30 + BLOCK_SIZE, 120:120 + BLOCK_SIZE].reshape(-1))
    validation_data.append(MyData(feature, torch.from_numpy(edge_index), vals))

for i in testing_indices:
    obs_reshaped = all_obs[i, valid_indices].reshape(-1, 1)
    feature = torch.from_numpy(np.concatenate((obs_reshaped, obs_station), axis=1))
    vals = torch.from_numpy(all_val[i, 30:30 + BLOCK_SIZE, 120:120 + BLOCK_SIZE].reshape(-1))
    testing_data.append(MyData(feature, torch.from_numpy(edge_index), vals))

# ======================== choose model ========================

model = None
if (obs.shape[1] == 94):
    model = model94()

if (obs.shape[1] == 415):
    model = model415()

if (obs.shape[1] == 19):
    model = model19()

# ======================== set nessesary components ========================

optimizer = optim.Adam(model.parameters(), lr=0.1)
# scheduler = StepLR(optimizer, step_size=50, gamma=0.7)
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

scheduler3 = StepLR(optimizer, step_size=200, gamma=0.7)

# ======================== traing and svae model ========================

loss_history = train(model, training_data, validation_data, optimizer, scheduler3, criterion)

# ======================== display results ========================

checkpoint = torch.load("checkpoints/test.pth", weights_only=True)
model.load_state_dict(checkpoint)

x, y = np.meshgrid(lat, lon, indexing='ij')
coordinate = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

total_error = np.zeros(BLOCK_SIZE ** 2)
for data in testing_data:
    predict = model(data)
    predict_val = predict.detach().numpy()
    real_val = data.vals.detach().numpy()

    error = (real_val - predict_val) ** 2
    total_error += error

avg_error = total_error / 30
print("Average mse: ", np.mean(avg_error))

plotLossCurve(loss_history)
plotError(coordinate, avg_error)
# plot3d(coordinate, real_val, predict_val)
# plot_compare_3d(coordinate, real_val, predict_val)

