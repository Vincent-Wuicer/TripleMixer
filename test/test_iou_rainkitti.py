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
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix



pred_dir = "/home/hit/code/WaffleIron/AMW-NET/result/semantic_rainkitti/predictions_semantic_rainkitti_32_9"
true_dir = "/home/hit/sda/Dataset/Weather_KITTI/Rain-KITTI"

def read_label_file(file_path):
    return np.fromfile(file_path, dtype=np.int32)


# 类别映射
label_to_class = {
    0: "not-rain",
    1: "rain",
    9 : "unlabelled"
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


# 使用类别索引初始化存储指标的字典，而不是类别名称
metrics = {cls_idx: {"precision": [], "recall": [], "iou": [], "f1": []} for cls_idx in [0, 1]}

for root, dirs, files in os.walk(pred_dir):
    for file in files:
        if file.endswith(".label"):
            pred_label_path = os.path.join(root, file)
            true_label_path = pred_label_path.replace("code/WaffleIron/AMW-NET/result/semantic_rainkitti/predictions_semantic_rainkitti_32_9", "sda/Dataset/Weather_KITTI/Rain-KITTI").replace("predictions", "rain_label")
            
            pred_label = read_label_file(pred_label_path)
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


# 使用类别名称打印每个类别的平均指标
for idx, cls_metrics in metrics.items():
    avg_precision = np.mean(cls_metrics["precision"])
    avg_recall = np.mean(cls_metrics["recall"])
    avg_iou = np.mean(cls_metrics["iou"])
    avg_f1 = np.mean(cls_metrics["f1"])
    
    cls_name = label_to_class[idx]  # 使用类别索引获取类别名称
    print(f"Class {cls_name} - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, IoU: {avg_iou:.4f}, F1: {avg_f1:.4f}")
