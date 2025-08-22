import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# 读取标签文件并计算特定标签的数量
file_name_kitti = '/home/hit/sda/Dataset/Weather_KITTI/Snow-KITTI/08/snow_labels/000055.label'
points_kitti = np.fromfile(file_name_kitti, dtype=np.int32)
count_ones = np.count_nonzero(points_kitti == 110)
print("The point label is:", count_ones)

unique_labels = np.unique(points_kitti)
print("Unique labels:", unique_labels)

# 读取点云文件
file_name = '/home/hit/sda/Dataset/Weather_KITTI/Snow-KITTI/08/snow_velodyne/000055.bin'
points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
print("The point is:", points.shape[0])

coords = points[:, :3]
intensities = points[:, 3]

# 将强度值归一化到范围 [0, 1]
normalized_intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
print("Intensity max is:", intensities.max())

# 使用Jet颜色映射
colormap = plt.get_cmap("jet")
colors = colormap(normalized_intensities)[:, :3]  # 获取RGB值

# 将标签为110的点设置为红色
colors[points_kitti == 110] = [1, 0, 0]  # 将标签为110的点设置为红色

# 创建并可视化点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 自定义可视化设置
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加点云并设置背景颜色
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])  # 设置背景为黑色
opt.point_size = 2  # 调整点大小以提高可见性

# 渲染可视化窗口
vis.run()
vis.destroy_window()
