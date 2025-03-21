from config import*
from mymodel import*
from plot import*
from train import*

# load data
lat = np.load("data/lat.npy").astype(np.float32)
lon = np.load("data/lon.npy").astype(np.float32)
real_vals = np.load("data/vals.npy").astype(np.float32)
obs_station = np.load("data/station.npy").astype(np.float32)
obs = np.load("data/obs.npy").astype(np.float32)
edge_index = np.load("data/edge_index.npy")

plotObs(lat, lon, obs_station)

training_data = []
for i in range(150):
    feature = torch.from_numpy(obs[i]).reshape(-1, 1)
    vals = torch.from_numpy(real_vals[i].reshape(-1))
    training_data.append(MyData(feature, torch.from_numpy(edge_index), vals))

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

start_time = time.time()
trainAndPlot(model, training_data, optimizer, criterion)
end_time = time.time()
print(f"gnn training time: {end_time - start_time}")

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
plot3d(coordinate, real_val, predict_val)
plot_compare_3d(coordinate, real_val, predict_val)

