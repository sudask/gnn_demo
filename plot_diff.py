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
    grid = generateGrid(size)
    val = H1(U(grid[:, 0], grid[:, 1])).unsqueeze(1)

    data = packData(grid, val, edge_index, coordinate)

    return data, coordinate, real_val


data, coordinate, real_val = generateData(GRID_SIZE)
predict_val = []
for i in range(len(data)):
    predict_val.append(model(data[i]))

predict_val = torch.stack(predict_val)

# 计算真实值与预测值的差值
diff_val = real_val - predict_val

# 找到差值中的最小值和最大值，用于统一颜色映射范围
min_val = torch.min(diff_val).item()
max_val = torch.max(diff_val).item()

coordinate_np = coordinate.detach().numpy()
diff_val_np = diff_val.detach().numpy()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# 设置统一的颜色映射范围
scatter = ax.scatter(coordinate_np[:, 0], coordinate_np[:, 1], c=diff_val_np, cmap='viridis', vmin=min_val, vmax=max_val)
ax.set_title('Difference between Real and Predict Values')
fig.colorbar(scatter, ax=ax)

plt.show()