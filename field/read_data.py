import numpy as np
import matplotlib.pyplot as plt
import torch
from mymodel import *
import torch.optim as optim
import random

lat = np.load("data/lat.npy").astype(np.float32)
lon = np.load("data/lon.npy").astype(np.float32)
real_vals = np.load("data/vals.npy").astype(np.float32)
obs_station = np.load("data/station.npy").astype(np.float32)
obs = np.load("data/obs.npy").astype(np.float32)
edge_index = np.load("data/edge_index.npy")

# plt.figure(figsize=(10, 8))
# plt.plot([lat[0], lat[-1], lat[-1], lat[0], lat[0]], [lon[0], lon[0], lon[-1], lon[-1], lon[0]], 'b-', label='Latitude-Longitude Range')
# plt.scatter(obs_station[:, 0], obs_station[:, 1], color='r', marker='o', label='Observation Stations')
# plt.xlabel('Latitude')
# plt.ylabel('Longitude')
# plt.title('Observation Stations within Latitude-Longitude Range')
# plt.legend()
# plt.grid(True)
# plt.show()

class MyData:
    def __init__(self, feature, edge_index, vals = None):
        self.feature = feature
        self.edge_index = edge_index
        self.vals = vals

print(real_vals[100])
exit()
training_data = []
for i in range(150):
    feature = torch.from_numpy(obs[i]).reshape(-1, 1)
    vals = torch.from_numpy(real_vals[i].reshape(-1))
    training_data.append(MyData(feature, torch.from_numpy(edge_index), vals))

testing_data = []
for i in range(150, 200):
    feature = torch.from_numpy(obs[i]).reshape(-1, 1)
    testing_data.append(MyData(feature, torch.from_numpy(edge_index)))

model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def trainAndPlot(model, training_data, optimizer, criterion):
    loss_history = []

    for epoch in range(200):
        random.shuffle(training_data)
        for data in training_data:
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, data.vals)

            loss.backward()
            optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 1 == 0:
            print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), "checkpoints/test.pth")

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

trainAndPlot(model, training_data, optimizer, criterion)
