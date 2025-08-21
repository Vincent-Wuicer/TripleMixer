<p align="center">
  <img src="Fig/tri.png" align="center" width="22.5%">
  
  <h3 align="center"><strong>TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather</strong></h3>

  <p align="center">
      <a href="https://scholar.google.com.sg/citations?user=miv8T6MAAAAJ&hl=zh-CN" target='_blank'>Xiongwei Zhao</a><sup>1*</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com.sg/citations?user=OTBgvCYAAAAJ&hl=zh-CN&oi=ao" target='_blank'>Congcong Wen</a><sup>2,3*</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com.sg/citations?user=VWjvfjkAAAAJ&hl=zh-CN" target='_blank'>Zhu Xu</a><sup>1#</sup>&nbsp;&nbsp;&nbsp;
      <a href="" target='_blank'>Yang Wang</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://www.researchgate.net/profile/Haojie-Bai" target='_blank'>Haojie Bai</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://scholar.google.com.sg/citations?user=WMyb00gAAAAJ&hl=zh-CN&oi=ao" target='_blank'>Wenhao Dou</a><sup>1</sup>
    <br>
  <sup>1</sup>Harbin Institute of Technology&nbsp;&nbsp;&nbsp;
  <sup>2</sup>Harvard University&nbsp;&nbsp;&nbsp;
  <sup>3</sup>New York University
  </p>

</p>


<p align="center">
  <a href="https://www.arxiv.org/pdf/2408.13802" target='_blank'>
    <img src="https://img.shields.io/badge/arXiv-2009.03137-b31b1b.svg">
  </a>
  
  <a href="https://github.com/Grandzxw/TripleMixer/stargazers" target='_blank'>
    <img src="https://img.shields.io/github/stars/Grandzxw/TripleMixer.svg">
  </a>

  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=Grandzxw.TripleMixer&left_color=gray&right_color=firebrick">
  </a>

</p>



## Abstract
Adverse weather conditions such as snow, fog, and rain pose significant challenges to LiDAR-based perception models by introducing noise and corrupting point cloud measurements. To address this issue, we make the following three contributions:
1. **Point cloud denoising network:** we propose **TripleMixer**, a robust and efficient point cloud denoising network that integrates spatial, frequency, and channel-wise processing through three specialized mixer modules. TripleMixer can be seamlessly deployed as a plug-and-play module within existing LiDAR perception pipelines;
2. **Large-scale adverse weather datasets:** we construct two large-scale simulated datasets, **Weather-KITTI** and **Weather-NuScenes**, covering diverse weather scenarios with dense point-wise semantic and noise annotations;
3. **LiDAR perception benchmarks:** we establish four benchmarks: **Denoising**, **Semantic Segmentation (SS)**, **Place Recognition (PR)**, and **Object Detection (OD)**. These benchmarks enable systematic evaluation of denoising generalization, transferability, and downstream impact under both simulated and real-world adverse weather conditions.


## Updates
* 08/21/2025: All codes and configurations have been updated!
* 12/26/2024: The Weather-KITTI and Weather-NuScenes datasets are publicly available on the BaiduPan platform!   
  - **Weather-KITTI:** [Download link](https://pan.baidu.com/s/1lwkIWwiLvtaM2SDKfT0SCg) (code: `xxr1`)  
  - **Weather-NuScenes:** [Download link](https://pan.baidu.com/s/1Qhr4I15W5IuamLC7gZTL8g) (code: `musq`)  
* 24/08/2024: Initial release and submitted to the Journal. The dataset will be open source soon!


## Dataset

### 1) Overview

Our **Weather-KITTI** and **Weather-NuScenes** are based on the [SemanticKITTI](https://www.semantic-kitti.org/) and [nuScenes-lidarseg](https://www.nuscenes.org/) datasets, respectively. These datasets cover three common adverse weather conditions: rain, fog, and snow and retain the original LiDAR acquisition information and provide point-level semantic labels for rain, fog, and snow. The visualization results are shown below:

<p align="center"> <img src="Fig/combined.png" width="50%" height="400px"> </p>

### 2) Dataset Statistics

<p align="center"> <img src="Fig/frames.png" width="95%"> </p>
<p align="center"> <img src="Fig/kitti_semantic.png" width="95%"> </p>


## Denoising network





[![arXiv](https://img.shields.io/badge/arXiv-2009.03137-b31b1b.svg)](https://www.arxiv.org/abs/2408.13802)
[![GitHub Stars](https://img.shields.io/github/stars/Grandzxw/TripleMixer.svg)](https://github.com/Grandzxw/TripleMixer/stargazers)

# TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather

This is the official repository of the **Weather-KITTI** and **Weather-NuScenes** dataset. For technical details, please refer to:

**TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather** <br />
[Xiongwei Zhao*](https://github.com/Grandzxw), [Congcong Wen*](https://scholar.google.com.sg/citations?user=OTBgvCYAAAAJ&hl=zh-CN&oi=ao), Yang Wang, 
[Haojie Bai](https://www.researchgate.net/profile/Haojie-Bai), [Wenhao Dou](https://scholar.google.com.sg/citations?user=WMyb00gAAAAJ&hl=zh-CN&oi=ao). <br />

**[[Paper](https://www.arxiv.org/abs/2408.13802)] [[Blog]()][[Download](https://pan.baidu.com/s/1lwkIWwiLvtaM2SDKfT0SCg)[code:xxr1]] ** <br />


### 1. Dataset

#### (1) Overview

In this Work, we propose our synthetic adverse weather datasets, named **Weather-KITTI** and **Weather-NuScenes**, which are based on the [SemanticKITTI](https://www.semantic-kitti.org/) and [nuScenes-lidarseg](https://www.nuscenes.org/) datasets, respectively. These datasets cover three common adverse weather conditions: rain, fog, and snow and retain the original LiDAR acquisition information and provide point-level semantic labels for rain, fog, and snow.

<p align="center"> <img src="Fig/combined.png" width="95%" height="700px"> </p>
<p align="center"> <img src="Fig/comparison.png" width="95%"> </p>


#### (2) Dataset Statistics

<p align="center"> <img src="Fig/frames.png" width="95%"> </p>

<table border="0">
  <tr>
    <td><img src="Fig/kitti_semantic.png" width="100%"></td>
    <td><img src="Fig/nus_semantic.png" width="100%"></td>
  </tr>
</table>


### 2. Quantitative Result of Denoising Task
<p align="center"> <img src="Fig/results.png" width="100%"> </p>


### 3. Training and Evaluation

Continuous Updates！


### Citation
If you find our work useful in your research, please consider citing:
```bibtex
@misc{zhao2024triplemixer3dpointcloud,
      title={TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather}, 
      author={Xiongwei Zhao and Congcong Wen and Yang Wang and Haojie Bai and Wenhao Dou},
      year={2024},
      eprint={2408.13802},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.13802}, 
}
```


### Updates
* 24/08/2024: Initial release and submitted to the Journal. The dataset will be open source soon!
* 12/26/2024: BaiduPan for Chinese mainland dataset download link has been updated!

### License
The dataset is based on the [SemanticKITTI](https://www.semantic-kitti.org/) dataset, provided under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States License (CC BY-NC-SA 3.0 US)](https://creativecommons.org/licenses/by-nc-sa/3.0/us/), and the [nuScenes-lidarseg](https://www.nuscenes.org/) dataset, provided under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). This dataset is provided under the terms of the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
