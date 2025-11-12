from config import *
from sklearn.neighbors import NearestNeighbors

def generateEdgeOfStation(station_pos):
    edge_index = []
    for i in range(len(station_pos)):
        for j in range(len(station_pos)):
            if np.linalg.norm(station_pos[i] - station_pos[j]) < 1.0:
                edge_index.append([i, j])
    
    return np.array(edge_index).T

def generateEdgeOfGrid(lat, lon, bias=0):
    edge_index = []
    
    n_lat = len(lat)
    n_lon = len(lon)
    
    for i in range(n_lat):
        for j in range(n_lon):
            cur_idx = i * n_lon + j
            
            # when append edge, you need to add bias
            if i > 0:
                neighbor_idx = (i - 1) * n_lon + j
                edge_index.append([bias + neighbor_idx, bias + cur_idx])
                edge_index.append([bias + cur_idx, bias + neighbor_idx])

            if i < n_lat - 1:
                neighbor_idx = (i + 1) * n_lon + j
                edge_index.append([bias + neighbor_idx, bias + cur_idx])
                edge_index.append([bias + cur_idx, bias + neighbor_idx])
                
            if j > 0:
                neighbor_idx = i * n_lon + j - 1
                edge_index.append([bias + neighbor_idx, bias + cur_idx])
                edge_index.append([bias + cur_idx, bias + neighbor_idx])
                
            if j < n_lon - 1:
                neighbor_idx = i * n_lon + j + 1
                edge_index.append([bias + neighbor_idx, bias + cur_idx])
                edge_index.append([bias + cur_idx, bias + neighbor_idx])
                
        
    return np.array(edge_index).T

def generateEdgeFromStation2Grid(station, lat, lon):
    k = 10
    n_lat = len(lat)
    n_lon = len(lon)
    
    # 创建网格点位置矩阵
    grid_points = []
    for i in range(n_lat):
        for j in range(n_lon):
            grid_points.append([lat[i], lon[j]])
    grid_points = np.array(grid_points)
    
    # 使用KNN找到每个网格点最近的k个站点
    nbrs = NearestNeighbors(n_neighbors=min(k, len(station)), algorithm='auto')
    nbrs.fit(station)
    distances, indices = nbrs.kneighbors(grid_points)
    
    # 构建边索引
    edge_index = []
    n_stations = len(station)
    
    for grid_idx, station_indices in enumerate(indices):
        for station_idx in station_indices:
            # 从站点指向网格点 (station_idx -> grid_idx + n_stations)
            edge_index.append([station_idx, n_stations + grid_idx])
    
    return np.array(edge_index).T
                
def generateEdgeIndex(obs_station_pos, target_point_lat, target_point_lon):
    edge_station = generateEdgeOfStation(obs_station_pos)
    edge_grid = generateEdgeOfGrid(target_point_lat, target_point_lon, len(obs_station_pos))
    edge_station_2_grid = generateEdgeFromStation2Grid(obs_station_pos, target_point_lat, target_point_lon)
    
    edge_index = np.concatenate((edge_station, edge_grid, edge_station_2_grid), axis=1)
    
    x, y = np.meshgrid(target_point_lat, target_point_lon, indexing='ij')
    target_points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    graph_nodes = np.concatenate((obs_station_pos, target_points), axis=0)
    
    return graph_nodes, edge_index
    


