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
    
    for seq in [str(i).zfill(2) for i in range(22)]:
        if seq not in ['07']:
            continue
        src_folder = os.path.join(args.root_folder, seq + '/velodyne')
        rain_txt = os.path.join(args.root_folder, seq + '/info_rain.txt')
        with open(rain_txt, 'r') as f:
            lines = f.readlines()
            rain_rate = float(lines[0].split('=')[1].strip())
        print("rain_rate is: ", rain_rate)
        

        all_files = []
        val_txt = os.path.join(args.root_folder, seq + '.txt')
        with open(val_txt, 'r') as f:
            for line in f.readlines():
                all_files.append(os.path.join(src_folder,line.strip()+'.bin'))
        
        all_paths =  copy.deepcopy(all_files)
        dst_folder = os.path.join(args.dst_folder, seq)
        Path(dst_folder).mkdir(parents=True, exist_ok=True)

        def _map(i: int) -> None:
            points = np.fromfile(all_paths[i], dtype=np.float32).reshape((-1, 4))
            label_path = all_paths[i].replace('Dataset/Ori_KITTI', 'SemanticKITTI/dataset').replace('velodyne', 'labels')[:-3] + 'label'
            labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
            labels = labels & 0xFFFF
            assert labels is not None
            
            # unique_labels = np.unique(labels)
            # print("Unique labels:", unique_labels)    
            rain_points, rain_semantic_labels = lisa.augment_mc(points, labels, rain_rate)
            rain_label = np.where(rain_points[:, -1] == 2, 0, rain_points[:, -1])
            rain_points = rain_points[:,:4]
            
            # print("fog_points is: ", fog_points.shape)
            # print("num_fog is: ", num_fog)
            # print("fog_label 1 is: ", np.sum(rain_label == 1))
            # print("fog_semanticlabel 112 is: ", np.sum(rain_semantic_labels == 112))

            lidar_save_path = os.path.join(dst_folder,'snow_velodyne', all_files[i].split('/')[-1])
            if not os.path.exists(os.path.dirname(lidar_save_path)):
                #os.makedirs(os.path.dirname(lidar_save_path))
                os.makedirs(os.path.dirname(lidar_save_path), exist_ok=True)
            
            label_save_path = os.path.join(dst_folder,'snow_label', all_files[i].split('/')[-1].replace('.bin', '.label'))
            if not os.path.exists(os.path.dirname(label_save_path)):
                #os.makedirs(os.path.dirname(label_save_path))
                os.makedirs(os.path.dirname(label_save_path), exist_ok=True)
            
            semanticlabel_save_path = os.path.join(dst_folder,'snow_labels', all_files[i].split('/')[-1].replace('.bin', '.label'))
            if not os.path.exists(os.path.dirname(semanticlabel_save_path)):
                #os.makedirs(os.path.dirname(label_save_path))
                os.makedirs(os.path.dirname(semanticlabel_save_path), exist_ok=True)    
            
            rain_points.astype(np.float32).tofile(lidar_save_path)
            rain_label.astype(np.int32).tofile(label_save_path)
            rain_semantic_labels.astype(np.int32).tofile(semanticlabel_save_path)

        n = len(all_files)
        with mp.Pool(args.n_cpus) as pool:
            l = list(tqdm(pool.imap(_map, range(n)), total=n))
    
