import os
import copy
import math
import argparse
import glob
from pathlib import Path
from atmos_models import LISA
import numpy as np
from os.path import basename, join, isdir
import time
import multiprocessing as mp
from tqdm import tqdm
from nuscenes import NuScenes
import pickle



if __name__ == '__main__':
        lisa = LISA(atm_model='rain')
        ## for the kitti
        ori_point_path = '/home/hit/sda/Dataset/Ori_KITTI/sequences/07/velodyne/000105.bin'
        #ori_point_path = '/home/hit/sda/Nuscenes/Nuscenes/samples/LIDAR_TOP/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883536948304.pcd.bin'
        ori_points = np.fromfile(ori_point_path, dtype=np.float32).reshape((-1, 4))
        
        label_path = ori_point_path.replace('Dataset/Ori_KITTI', 'SemanticKITTI/dataset').replace('velodyne', 'labels')[:-3] + 'label'
        #label_path = '/home/hit/sda/Nuscenes/Nuscenes/lidarseg/v1.0-trainval/d001097c1fd340f3aad2cca489bec144_lidarseg.bin'
        print("label_path is: ", label_path)
        #print("label_path is: ", label_path)
        labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
        labels = labels & 0xFFFF
        assert labels is not None
        
        rain_rate = 120  # 100 200 300 400 500
        # unique_labels = np.unique(labels)
        # print("Unique labels:", unique_labels)    
        rain_points, rain_semantic_labels = lisa.augment_mc(ori_points, labels, rain_rate)
        rain_label = np.where(rain_points[:, -1] == 2, 0, rain_points[:, -1])
        rain_points = rain_points[:,:4]
        
        # print("fog_points is: ", fog_points.shape)
        # print("num_fog is: ", num_fog)
        # print("fog_label 1 is: ", np.sum(rain_label == 1))
        # print("fog_semanticlabel 112 is: ", np.sum(rain_semantic_labels == 112))
        dst_folder = '/home/hit/sda/Simu_tool/rain_kitti_v1/06'

        lidar_save_path = os.path.join(dst_folder,'rain_velodyne', ori_point_path.split('/')[-1])
        if not os.path.exists(os.path.dirname(lidar_save_path)):
            #os.makedirs(os.path.dirname(lidar_save_path))
            os.makedirs(os.path.dirname(lidar_save_path), exist_ok=True)
        
        # label_save_path = os.path.join(dst_folder,'rain_label', all_files[i].split('/')[-1].replace('.bin', '.label'))
        # if not os.path.exists(os.path.dirname(label_save_path)):
        #     #os.makedirs(os.path.dirname(label_save_path))
        #     os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
        
        semanticlabel_save_path = os.path.join(dst_folder,'rain_labels', ori_point_path.split('/')[-1].replace('.bin', '.label'))
        if not os.path.exists(os.path.dirname(semanticlabel_save_path)):
            #os.makedirs(os.path.dirname(label_save_path))
            os.makedirs(os.path.dirname(semanticlabel_save_path), exist_ok=True)    
        
        print(semanticlabel_save_path)
        rain_points.astype(np.float32).tofile(lidar_save_path)
        rain_semantic_labels.astype(np.int32).tofile(semanticlabel_save_path)