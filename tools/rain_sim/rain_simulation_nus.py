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



def parse_arguments():
    parser = argparse.ArgumentParser(description='LiDAR raingification')
    parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default= mp.cpu_count())
    parser.add_argument('-r', '--root_folder', help='root folder of dataset', type=str,
                        default='./data_root/Kitti/')
    parser.add_argument('-d', '--dst_folder', help='savefolder of dataset', type=str,
                        default='./save_root/fog/light')
    arguments = parser.parse_args()
    return arguments




if __name__ == '__main__':
    args = parse_arguments()
    print('')
    print(f'using {args.n_cpus} CPUs')
    
    lisa = LISA(atm_model='rain')
    
    all_files = []
    print("root is: ", args.root_folder)
    nusc_info = NuScenes(version='v1.0-trainval', dataroot=args.root_folder, verbose=False)
    imageset = os.path.join(args.root_folder,"nuscenes_infos_train.pkl")
    print(imageset)
    with open(imageset, 'rb') as f:
            infos = pickle.load(f)
    all_files = infos['infos']
    # print("all_files", all_files[0])
    # print("all_files", all_files[10])
    all_paths =  copy.deepcopy(all_files)
    #print("all_paths is: ", all_paths)
    num_splits = 10
    split_size = len(all_files) // num_splits
    print("split size is:", split_size)
    
    splits_all_files = [all_files[i:i + split_size] for i in range(0, len(all_files), split_size)]
    splits_all_paths = [all_paths[i:i + split_size] for i in range(0, len(all_paths), split_size)]
    dst_folder = args.dst_folder
    
    print("all_files len is: ",len(all_files))

    for idx, (split_files, split_paths) in enumerate(zip(splits_all_files, splits_all_paths)):
        assert len(split_files) == len(split_paths)
        print("idx is: ", idx)
        print("len of split_files is: ",len(split_files))
        
        if idx == 0:
            rain_rate = 500.0
        elif idx == 1:
            rain_rate = 1000.0
        elif idx == 2:
            rain_rate = 1500.0
        elif idx == 3:
            rain_rate = 2000.0 
        elif idx == 4:
            rain_rate = 2500.0  
        elif idx == 5:
            rain_rate = 2000.0  
        elif idx == 6:
            rain_rate = 1500.0  
        elif idx == 7:
            rain_rate = 500.0  
        elif idx == 8:
            rain_rate = 2500.0  
        elif idx == 9:
            rain_rate = 1000.0  
        else:
            continue
    
    
        #print("rain_rate is: ", rain_rate)
        
        # root
        num_str = str(idx).zfill(2)
        Path(dst_folder).mkdir(parents=True, exist_ok=True)
        lidar_save_root = os.path.join(dst_folder, num_str, 'rain_velodyne')
        if not os.path.exists(lidar_save_root):
            os.makedirs(lidar_save_root)

        snow_label_root = os.path.join(dst_folder, num_str, 'rain_label')
        if not os.path.exists(snow_label_root):
            os.makedirs(snow_label_root)

        label_save_root = os.path.join(dst_folder,  num_str, 'rain_labels')
        if not os.path.exists(label_save_root):
            os.makedirs(label_save_root)


        def _map(i: int) -> None:
            info = split_paths[i]
            lidar_path = info['lidar_path'][16:]
            lidar_sd_token = nusc_info.get('sample', info['token'])['data']['LIDAR_TOP']
            #print("lidar_sd_token len is: ",lidar_sd_token)
            file_path = os.path.join(args.root_folder, lidar_path)
            if os.path.exists(file_path):
                ori_points = np.fromfile(file_path, dtype=np.float32, count=-1).reshape([-1, 5])
                points = ori_points[:,:4]
                #print("points shape is:", points.shape)
                #print("file_path is: ", file_path)
                label_path = nusc_info.get('lidarseg', lidar_sd_token)['filename']
                #print("label path is: ", label_path)
                lidarseg_labels_filename = os.path.join(args.root_folder, label_path)
                #print("lidarseg_labels_filename",lidarseg_labels_filename)
                label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
                labels = label & 0xFFFF
                assert labels is not None 
                unique_labels = np.unique(labels)
                #print("Unique labels:", unique_labels)
                
                rain_points, rain_semantic_labels = lisa.augment_mc(points, labels, rain_rate)
                rain_label = np.where(rain_points[:, -1] == 2, 0, rain_points[:, -1])
                rain_points = rain_points[:,:4]
                # print("fog_points is: ", fog_points.shape)
                # print("num_fog is: ", num_fog)
                print("rain_label 1 is: ", np.sum(rain_label == 1))
                # print("fog_semanticlabel 112 is: ", np.sum(rain_semantic_labels == 112)) 
                lidar_filename = os.path.basename(lidar_path)
                
                lidar_save_path = os.path.join(lidar_save_root, lidar_filename)
                label_save_path = os.path.join(snow_label_root, lidar_filename.replace('.bin', '.label'))
                semanticlabel_save_path = os.path.join(label_save_root, lidar_filename.replace('.bin', '.label'))

                rain_points.astype(np.float32).tofile(lidar_save_path)
                rain_label.astype(np.int32).tofile(label_save_path)
                rain_semantic_labels.astype(np.int32).tofile(semanticlabel_save_path)


        n = len(split_paths)
        with mp.Pool(args.n_cpus) as pool:
            l = list(tqdm(pool.imap(_map, range(n)), total=n))
    
