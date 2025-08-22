# Copyright 2024 - xiongwei zhao @ grandzhaoxw@gmail.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import glob
import os


color_map = {
    10: [245, 150, 100],
    11: [245, 230, 100],
    0: [250, 80, 100],
    1: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [255, 255, 255],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0]
}




def read_labels(label_file):
    label = np.fromfile(label_file, dtype=np.uint32)
    label = label.reshape((-1))
    upper_half = label >> 16      # get upper half for instances
    lower_half = label & 0xFFFF   # get lower half for semantics
    return lower_half

def to_XYZI_array(points_struct):
    points = np.zeros((points_struct['x'].shape[0], 4), dtype=float)
    points[:, 0] = points_struct['x']
    points[:, 1] = points_struct['y']
    points[:, 2] = points_struct['z']
    # Load intensity
    if 'intensity' in points_struct.dtype.names:
        points[:, 3] = points_struct['intensity']
    elif 'i' in points_struct.dtype.names:
        points[:, 3] = points_struct['intensity']
    else:
        print("intensity not found, that's probably not good but feel free to supress this")
    return points

label_in = '/home/hit/sda/WADS2/sequences/11/snow_labels'
sequence_pcd_in = '/home/hit/sda/WADS2/sequences/11/velodyne'

label_list = glob.glob(label_in + '/*')

for f, label_fp in enumerate(label_list):
    if f > 2:
        break
    print('{}/{}'.format(f, len(label_list)))
    plt.style.use('dark_background')
    idx = os.path.basename(label_fp).split('.')[0] + '.' + os.path.basename(label_fp).split('.')[1]
    # BIN KITTI FILES
    lidar_fp = os.path.join(sequence_pcd_in, idx[:-6] + '.bin')
    lidar = np.fromfile(lidar_fp, dtype=np.float32).reshape(-1, 4)

    label = read_labels(label_fp)

    fig, ax = plt.subplots(figsize=(16, 16))
    # plt.set
    # plt.figure(num=1, figsize=(16, 16), dpi=600, facecolor='w', edgecolor='k')
    colors = [[color_map[i][0]/255, color_map[i][1]/255, color_map[i][2]/255] for i in label]
    ax.scatter(lidar[:, 0],
               lidar[:, 1],
               s=0.5, c=colors, marker='o', facecolor=colors)
    ax.axis('square')
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_axis_off()
    plt.show()


    #plt.savefig(os.path.join('/home/map4/pvkd/out_cyl/semantickitti_multiscan10_dyn_fig', idx+'.png'), dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()