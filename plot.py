from model import *
from data_prep import *
import matplotlib.pyplot as plt
import torch


model = SimpleGNN()
model_path = "./checkpoints/trained_model_2.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))


def generateData(size, num_points=10000, left_end=0, right_end=10):
    x = torch.rand(num_points) * (right_end - left_end) + left_end
    y = torch.rand(num_points) * (right_end - left_end) + left_end

    real_val = U(x, y)
    coordinate = torch.stack((x, y), dim=1)
    edge_index = generateEdgeIndex(size)
    grid, val = generateGrid(size)

    data = packData(grid, val, edge_index, coordinate)

    return data, coordinate, real_val


data, coordinate, real_val = generateData(GRID_SIZE)
predict_val = []
for i in range(len(data)):
    predict_val.append(model(data[i]))

predict_val = torch.stack(predict_val)

# 找到真实值和预测值中的最小值和最大值，用于统一颜色映射范围
min_val = min(torch.min(real_val).item(), torch.min(predict_val).item())
max_val = max(torch.max(real_val).item(), torch.max(predict_val).item())

coordinate_np = coordinate.detach().numpy()
real_val_np = real_val.detach().numpy()
predict_val_np = predict_val.detach().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 设置统一的颜色映射范围
scatter1 = ax1.scatter(coordinate_np[:, 0], coordinate_np[:, 1], c=real_val_np, cmap='viridis', vmin=min_val, vmax=max_val)
ax1.set_title('Real Values')
fig.colorbar(scatter1, ax=ax1)

scatter2 = ax2.scatter(coordinate_np[:, 0], coordinate_np[:, 1], c=predict_val_np, cmap='viridis', vmin=min_val, vmax=max_val)
ax2.set_title('Predict Values')
fig.colorbar(scatter2, ax=ax2)

plt.show()