from config import*
from mymodel import*
from plot import*
from train import*
from read_and_save_ncdata import*

BLOCK_SIZE = 80 # differenct size should use different model
NUM_TRAINING_DATA = 150

# prepare data
if not os.path.exists(f"data/lat_{BLOCK_SIZE}.npy"):
    prepareNcdata(BLOCK_SIZE)

# load data
lat = np.load(f"data/lat_{BLOCK_SIZE}.npy").astype(np.float32)
lon = np.load(f"data/lon_{BLOCK_SIZE}.npy").astype(np.float32)
real_vals = np.load(f"data/vals_{BLOCK_SIZE}.npy").astype(np.float32)
obs_station = np.load(f"data/station_{BLOCK_SIZE}.npy").astype(np.float32)
obs = np.load(f"data/obs_{BLOCK_SIZE}.npy").astype(np.float32)
edge_index = np.load(f"data/edge_index_{BLOCK_SIZE}.npy")

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

feature = torch.from_numpy(obs[160]).reshape(-1, 1)
testing_data = MyData(feature, torch.from_numpy(edge_index))

predict = model(testing_data)
predict_val = predict.detach().numpy()
x, y = np.meshgrid(lat, lon, indexing='ij')
coordinate = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
print(coordinate.shape)
real_val = real_vals[160].reshape(-1)

plotDiff(coordinate, real_val, predict_val)
# plot3d(coordinate, real_val, predict_val)
# plot_compare_3d(coordinate, real_val, predict_val)

