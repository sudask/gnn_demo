import sys
sys.path.append("..")
from model import *
from data_prep import *
import matplotlib.pyplot as plt

model = SimpleGNN()
model_path = "../checkpoints/trained_model.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))

def generateData(size, num_points=1000, left_end=0, right_end=10):
    x = torch.rand(num_points) * (right_end - left_end) + left_end
    y = torch.rand(num_points) * (right_end - left_end) + left_end

    val = U(x, y)
    coordinate = torch.stack((x, y), dim=1)
    edge_index = generateEdgeIndex(size)
    feature = generateFeature(size)
    
    data = packData(feature, edge_index, coordinate)

    return data, coordinate, val

data, coordinate, real_val = generateData(10)
predict_val = []
for i in range(len(data)):
    predict_val.append(model(data[i]))

predict_val = torch.stack(predict_val)

coordinate_np = coordinate.detach().numpy()
real_val_np = real_val.detach().numpy()
predict_val_np = predict_val.detach().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

scatter1 = ax1.scatter(coordinate_np[:, 0], coordinate_np[:, 1], c=real_val_np, cmap='viridis')
ax1.set_title('Real Values')
fig.colorbar(scatter1, ax=ax1)

scatter2 = ax2.scatter(coordinate_np[:, 0], coordinate_np[:, 1], c=predict_val_np, cmap='viridis')
ax2.set_title('Predict Values')
fig.colorbar(scatter2, ax=ax2)

plt.show()