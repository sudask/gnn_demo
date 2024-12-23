import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def U(x, y):
    term1 = np.sin(x + np.cos(x + y))
    term2 = 0.5 * np.cos(x / 2 - y / 3) * np.sin(y / 2 - x / 2)
    term3 = 0.1 * np.sin(2 * x - 3 * y) * np.cos(4 * y - 3 * x)
    return term1 + term2 - term3

def H1(u):
    return u ** 2 - u

def H2(u):
    return np.cos(u - np.sin(u))

x = np.linspace(0.2, 10, 50)
y = np.linspace(0.2, 10, 50)
x, y = np.meshgrid(x, y, indexing='ij')
u_original = U(x, y)
obs = H1(u_original)

u = np.linspace(-2, 2, 1000)

y1 = H1(u)
y2 = H2(u)

plt.figure()  # 新起一个图形
plt.plot(u, y1, label='H1(u) = u**2 - u', color='blue')
plt.plot(u, y2, label='H2(u) = cos(u - sin(u))', color='red')
plt.title('Function Plots')
plt.xlabel('u')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')  # 121表示1行2列的第1个子图
# 将二维网格数据拉平为一维数组，用于散点图绘制（因为散点图要求数据是一维形式的坐标）
x_flat = x.flatten()
y_flat = y.flatten()
u_original_flat = u_original.flatten()
ax1.scatter(x_flat, y_flat, u_original_flat, c='b', marker='o', label='U')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('U')
ax1.set_title('3D Plot of U')
ax1.legend()

# 绘制第二个3D散点图（第二个子图）
ax2 = fig.add_subplot(122, projection='3d')  # 122表示1行2列的第2个子图
obs_flat = obs.flatten()
ax2.scatter(x_flat, y_flat, obs_flat, c='r', marker='s', label='Obs')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Obs')
ax2.set_title('3D Plot of Obs')
ax2.legend()

plt.show()

x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
x, y = np.meshgrid(x, y, indexing='ij')
u_original = U(x, y)
obs = H1(u_original)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
contour_original = plt.contour(x, y, u_original)
plt.clabel(contour_original, inline=True, fontsize=8)
plt.title("Original Function u_original")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 2, 2)
contour_obs = plt.contour(x, y, obs)
plt.clabel(contour_obs, inline=True, fontsize=8)
plt.title("Transformed Function obs (after H1)")
plt.xlabel("x")
plt.ylabel("y")

levels = contour_obs.levels
# 找到所有level为0的等高线索引
level_0_indices = np.where(levels == 0)[0]
index = 16
all_segments = contour_obs.allsegs[1]
segment = all_segments[index]
all_points = segment
selected_indices = np.random.choice(len(all_points), 8, replace=False)
selected_points = all_points[selected_indices]

# 在u_original对应的等高线图上标记点
plt.subplot(1, 2, 1)
plt.scatter(selected_points[:, 0], selected_points[:, 1], c='r', marker='o', label='Selected Points')
plt.legend()

# 在obs对应的等高线图上标记同样的点（注意正如你说大概率不在同一条等高线上）
plt.subplot(1, 2, 2)
plt.scatter(selected_points[:, 0], selected_points[:, 1], c='r', marker='o', label='Selected Points')
plt.legend()

plt.tight_layout()
plt.show()