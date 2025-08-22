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

import os
import yaml
import torch
import argparse
import network
import numpy as np
from tqdm import tqdm
from network import Triplemixer
from datasets import SnowNus, Collate
from sklearn.metrics import multilabel_confusion_matrix


def read_label_file(file_path):
    return np.fromfile(file_path, dtype=np.int32)


# 类别映射
label_to_class = {
    0: "not-snow",
    1: "snow"
}


def calculate_metrics(cm):
    """从混淆矩阵计算指标"""
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, iou, f1


if __name__ == "__main__":
    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint")
    parser.add_argument(
        "--path_dataset", type=str, help="Path to SemanticKITTI dataset"
    )
    parser.add_argument("--result_folder", type=str, help="Path to where result folder")
    parser.add_argument(
        "--num_votes", type=int, default=1, help="Number of test time augmentations"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--phase", required=True, help="val or test")
    args = parser.parse_args()
    assert args.num_votes % args.batch_size == 0
    os.makedirs(args.result_folder, exist_ok=True)

    # --- Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- SemanticKITTI (from https://github.com/PRBonn/semantic-kitti-api/blob/master/remap_semantic_labels.py)
    with open("./datasets/snow-nus.yaml") as stream:
        semkittiyaml = yaml.safe_load(stream)
    remapdict = semkittiyaml["learning_map_inv"]
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())


    # --- Dataloader
    tta = args.num_votes > 1
    dataset = SnowNus(
        rootdir=args.path_dataset,
        input_feat=config["geometry"]["input_feat"],
        voxel_size=config["geometry"]["voxel_size"],
        num_neighbors=config["geometry"]["neighbors"],
        dim_proj=config["resolution"]["dim_proj"],
        grids_shape=config["resolution"]["grids_size"],
        fov_xyz=config["resolution"]["fov_xyz"],
        phase=args.phase,
        tta=tta,
    )
    if args.num_votes > 1:
        new_list = []
        for f in dataset.im_idx:
            for v in range(args.num_votes):
                new_list.append(f)
        dataset.im_idx = new_list
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=Collate(),
    )
    args.num_votes = args.num_votes // args.batch_size


    # --- Build network
    net = Triplemixer(
        input_channels=config["geometry"]["size_input"],
        feat_channels=config["resolution"]["nb_channels"],
        depth=config["resolution"]["depth"],
        grid_shape=config["resolution"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
        drop_path_prob=config["resolution"]["drop"],
    )
    net = net.cuda()

    # --- Load weights
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    try:
        net.load_state_dict(ckpt["net"])
    except:
        # If model was trained using DataParallel or DistributedDataParallel
        state_dict = {}
        for key in ckpt["net"].keys():
            state_dict[key[len("module."):]] = ckpt["net"][key]
        net.load_state_dict(state_dict)
    #net.compress()
    net.eval()

    # --- Re-activate droppath if voting
    if tta:
        for m in net.modules():
            if isinstance(m, network.mixer.DropPath):
                m.train()

    metrics = {cls_idx: {"precision": [], "recall": [], "iou": [], "f1": []} for cls_idx in [0, 1]}

    # --- Evaluation
    id_vote = 0
    for it, batch in enumerate(
        tqdm(loader, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")
    ):
        # Reset vote
        if id_vote == 0:
            vote = None

        # Network inputs
        feat = batch["feat"].cuda(non_blocking=True)
        labels = batch["labels_orig"].cuda(non_blocking=True)
        batch["upsample"] = [up.cuda(non_blocking=True) for up in batch["upsample"]]
        cell_ind = batch["cell_ind"].cuda(non_blocking=True)
        occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
        neighbors_emb = batch["neighbors_emb"].cuda(non_blocking=True)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # Get prediction
        with torch.autocast("cuda", enabled=True):
            with torch.inference_mode():
                # Get prediction
                out = net(*net_inputs)
                for b in range(out.shape[0]):
                    temp = out[b, :, batch["upsample"][b]].T
                    if vote is None:
                        vote = torch.softmax(temp, dim=1)
                    else:
                        vote += torch.softmax(temp, dim=1)
        id_vote += 1

        # Save prediction
        if id_vote == args.num_votes:
            # Convert label
            pred_label = (
                vote.max(1)[1] + 1
            )  # Shift by 1 because of ignore_label at index 0
            label = pred_label.cpu().numpy().reshape(-1).astype(np.uint32)
            upper_half = label >> 16  # get upper half for instances
            lower_half = label & 0xFFFF  # get lower half for semantics
            lower_half = remap_lut[lower_half]  # do the remapping of semantics
            label = (upper_half << 16) + lower_half  # reconstruct full label
            label = label.astype(np.uint32)
            # Save result
            assert batch["filename"][0] == batch["filename"][-1]
            label_file = batch["filename"][0][
                len(os.path.join(dataset.rootdir)):
            ]
            label_file = label_file.replace("snow_velodyne", "predictions")[:-3] + "label"
            label_file = os.path.join(args.result_folder, label_file.lstrip('/'))
            
            os.makedirs(os.path.split(label_file)[0], exist_ok=True)
            label.tofile(label_file)
            
            absolute_path = os.path.abspath(label_file)
            
            #print("label_file is: ", absolute_path)
            
            true_label_path = absolute_path.replace("code/WaffleIron/AMW-NET/result/semantic_snownus/predictions_snownus_32_9", "sda/Dataset/Weather_Nuscenes/Snow-Nuscenes").replace("predictions", "snow_label")
            pred_label = read_label_file(absolute_path)
            true_label = read_label_file(true_label_path)
            
            print("pred_label is 0 : ", np.sum(pred_label == 0))
            print("true_label is 0 : ", np.sum(true_label == 0))
            
            print("pred_label is 1 : ", np.sum(pred_label == 1))
            print("true_label is 1 : ", np.sum(true_label == 1))

            cm = multilabel_confusion_matrix(true_label, pred_label, labels=[0, 1])
            
            for idx, cls_cm in enumerate(cm):
                precision, recall, iou, f1 = calculate_metrics(cls_cm)
                
                # 直接使用类别索引更新metrics字典
                metrics[idx]["precision"].append(precision)
                metrics[idx]["recall"].append(recall)
                metrics[idx]["iou"].append(iou)
                metrics[idx]["f1"].append(f1)

            # Reset count of votes
            id_vote = 0


    for idx, cls_metrics in metrics.items():
        avg_precision = np.mean(cls_metrics["precision"])
        avg_recall = np.mean(cls_metrics["recall"])
        avg_iou = np.mean(cls_metrics["iou"])
        avg_f1 = np.mean(cls_metrics["f1"])
        
        cls_name = label_to_class[idx]  # 使用类别索引获取类别名称
        print(f"Class {cls_name} - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, IoU: {avg_iou:.4f}, F1: {avg_f1:.4f}")


### 看下使用TTA的情况