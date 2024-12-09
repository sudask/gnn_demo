from config import *

def plotDiff(coordinate, real_val, predict_val):
    diff_val = real_val - predict_val

    min_val = np.min(diff_val)
    max_val = np.max(diff_val)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    scatter = ax.scatter(coordinate[:, 0], coordinate[:, 1], c=diff_val, cmap='viridis', vmin=min_val, vmax=max_val)
    ax.set_title('Difference between Real and Predict Values')
    fig.colorbar(scatter, ax=ax)

    plt.show()

def plot(coordinate, real_val, predict_val):
    min_val = min(np.min(real_val), np.min(predict_val))
    max_val = max(np.min(real_val), np.max(predict_val))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    scatter1 = ax1.scatter(coordinate[:, 0], coordinate[:, 1], c=real_val, cmap='viridis', vmin=min_val, vmax=max_val)
    ax1.set_title('Real Values')
    fig.colorbar(scatter1, ax=ax1)

    scatter2 = ax2.scatter(coordinate[:, 0], coordinate[:, 1], c=predict_val, cmap='viridis', vmin=min_val, vmax=max_val)
    ax2.set_title('Predict Values')
    fig.colorbar(scatter2, ax=ax2)

    plt.show()
    
from mpl_toolkits.mplot3d import Axes3D

def plot3d(coordinate, real_val, predict_val):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coordinate[:, 0], coordinate[:, 1], real_val, c='blue', label='Real Values', alpha=0.6)
    ax.scatter(coordinate[:, 0], coordinate[:, 1], predict_val, c='red', label='Predict Values', alpha=0.6)

    ax.set_title('Real and Predict Values in 3D')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()