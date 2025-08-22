import os
import numpy as np
import open3d as o3d
import multiprocessing
import matplotlib.pyplot as plt


# /home/hit/code/lisnownet-main/data/wads/11/velodyne/039498.bin
# /home/hit/Datesets/snowyKITTI/dataset/sequences/01/snow_velodyne/000000.bin
# /home/hit/sdb1/Dataset/cadcd/2018_03_06/0001/raw/lidar_points_corrected/data/0000000001.bin

# /home/hit/code/lisnownet-main/data/wads/11/labels/039498.label
# /home/hit/Datesets/snowyKITTI/dataset/sequences/00/snow_labels/000000.label
 
# file_name_wads = '/home/hit/code/lisnownet-main/data/wads/11/labels/039498.label'
# points_label = np.fromfile(file_name_wads, dtype=np.int32)
# count = np.count_nonzero(points_label == 110)
# print("Number of points with label 110:", count)

# file_name = '/home/hit/code/lisnownet-main/data/wads/11/velodyne/039498.bin'
# points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
# count_point = np.count_nonzero(points[:, 3] == 0.00000000)

# print("Number of points with last column as 0.000:", count_point)
# filtered_points = points[points_label == 110]
# last_column = filtered_points[:, 3]
# np.savetxt('/home/hit/code/lisnownet-main/util/last_column_points_label_110.txt', last_column, fmt='%.8f')

# /home/hit/sda/Nuscenes/snow/09/samples/LIDAR_TOP/snow_label/n015-2018-11-14-18-57-54+0800__LIDAR_TOP__1542193524298942.pcd.label
# /home/hit/sda/Nuscenes/snow/09/samples/LIDAR_TOP/snow_velodyne/n015-2018-11-14-18-57-54+0800__LIDAR_TOP__1542193524298942.pcd.bin

# /home/hit/sda/Dataset/Nuscenes/Snow-Nuscenes/00/snow_velodyne/n015-2018-07-18-11-50-34+0800__LIDAR_TOP__1531886001048259.pcd.bin
# n015-2018-07-24-11-13-19+0800__LIDAR_TOP__1532402309697154.pcd.label

# /home/hit/sda/Dataset/Snow_KITTI/sequences/01/snow_velodyne/000055.bin     snow_labels
# /home/hit/sda/Dataset/Snow_KITTI/sequences/01/snow_labels/000055.label

file_name_kitti = '/home/hit/sda/Dataset/Snow_KITTI/sequences/01/snow_labels/000055.label'
points_kitti = np.fromfile(file_name_kitti, dtype=np.int32)
count_ones = np.count_nonzero(points_kitti == 1)
print("The point label is:", count_ones)

unique_labels = np.unique(points_kitti)
print("Unique labels:", unique_labels)


file_name = '/home/hit/sda/Dataset/Snow_KITTI/sequences/01/snow_velodyne/000055.bin'
points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
print("The point is:", points.shape[0])

coords = points[:, :3]
intensities = points[:, 3]

# Normalize the intensity values to the range [0, 1]      intensities / 255.0
normalized_intensities =  intensities  #intensities - intensities.min()) / (intensities.max() - intensities.min())
print("intensity max is: ", intensities.max())

colormap = plt.get_cmap("jet")
colors = colormap(normalized_intensities)[:, :3]  # Get RGB values

# Create and visualize the point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(coords)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Custom visualization settings
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add point cloud and set background color
vis.add_geometry(pcd)
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])  # Set background to black
opt.point_size = 2  # Adjust point size for better visibility

# Render the visualizer
vis.run()
vis.destroy_window()






#np.savetxt('/home/hit/Datesets/snowyKITTI/dataset/sequences/00/000000_bin.txt', points,fmt='%.8f')
# points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)


# mask = points[:, 3] != 0.00000000  
# filtered_points = points[mask]  


# filtered_points.tofile('/home/hit/code/lisnownet-main/data/039498.bin') 


#np.savetxt('/home/hit/sdb1/Dataset/cadcd/0000000001.txt', points,fmt='%.8f')
#np.savetxt('/home/hit/code/lisnownet-main/util/kitti_labels.txt', points_kitti,fmt='%.8f')

