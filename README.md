
# HANet-Change-Detection :https://chengxihan.github.io/
You can still run HANet in the open-cd repository.  https://github.com/likyoo/open-cd

The Pytorch implementation for::gift::gift::gift:
“[HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/10093022),” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.
 C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, yum::yum::yum:


 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-googlegz-cd)](https://paperswithcode.com/sota/change-detection-on-googlegz-cd?p=hanet-a-hierarchical-attention-network-for)

 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-sysu-cd)](https://paperswithcode.com/sota/change-detection-on-sysu-cd?p=hanet-a-hierarchical-attention-network-for)

 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-levir)](https://paperswithcode.com/sota/change-detection-on-levir?p=hanet-a-hierarchical-attention-network-for)

 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-dsifn-cd)](https://paperswithcode.com/sota/change-detection-on-dsifn-cd?p=hanet-a-hierarchical-attention-network-for)


 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-whu-cd)](https://paperswithcode.com/sota/change-detection-on-whu-cd?p=hanet-a-hierarchical-attention-network-for)

 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-s2looking)](https://paperswithcode.com/sota/change-detection-on-s2looking?p=hanet-a-hierarchical-attention-network-for)

 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-cdd-dataset-season-1)](https://paperswithcode.com/sota/change-detection-on-cdd-dataset-season-1?p=hanet-a-hierarchical-attention-network-for)

 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/hanet-a-hierarchical-attention-network-for/change-detection-on-levir-cd)](https://paperswithcode.com/sota/change-detection-on-levir-cd?p=hanet-a-hierarchical-attention-network-for)
 

[14 Aril. 2023] Release the first version of the HANet
![image-20230415](/picture/HANet.png)

### Requirement  
```bash
-Pytorch 1.8.0  
-torchvision 0.9.0  
-python 3.8  
-opencv-python  4.5.3.56  
-tensorboardx 2.4  
-Cuda 11.3.1  
-Cudnn 11.3  
```


## Revised parameters 
 You can revise related parameters in the `metadata.json` file.  
 
## Training, Test and Visualization Process   

```bash
python trainHCX.py 
python test.py 
python Output_Results.py
```

## Test our trained model result  
You can directly test our model by our provided training weights in  `tmp/WHU, LEVIR, SYSU, and S2Looking `. And make sure the weight name is right. Of course, for different datasets, the `Dataset mean and std setting` is different.
```bash
path = opt.weight_dir+'final_epoch99.pt'
```

## Dataset Download   
 LEVIR-CD：https://justchenhao.github.io/LEVIR/  
 
 WHU-CD：http://gpcv.whu.edu.cn/data/building_dataset.html ,our paper split in [Baidu Disk](https://pan.baidu.com/s/16g3H1UsDMgqmXaVjiE319Q?pwd=6969),pwd:6969
 
SYSU-CD: Our paper split in [Baidu Disk](https://pan.baidu.com/s/1p0QfogZm4BM0dd1a0LTBBw?pwd=2023),pwd:2023

S2Looking-CD: Our paper split in [Baidu Disk](https://pan.baidu.com/s/1wAXPHhCLJTqPX0pC2RBMsg?pwd=2023),pwd:2023

CDD-CD: Our split in [Baidu Disk](https://pan.baidu.com/s/1cwJ0mEhcrbCWOJn5n-N5Jw?pwd=2023),pwd:2023

DSIFN-CD: Our split in [Baidu Disk]( https://pan.baidu.com/s/1-GD3z_eMoQglSJoi9P-6gw?pwd=2023),pwd:2023

 Note: Please crop the LEVIR dataset to a slice of 256×256 before training with it.
 ![image-20230415](/picture/HANet-WHU-LEVIR.png)
 ![image-20230415](/picture/HANet-ExperimentResult.png)
 
 And also we provide all test results of our HANet in the HANetTestResult!!!! Download in HANetTestResult or [Baidu Disk](https://pan.baidu.com/s/1nwPYkqtUIKe90KZoT5VO-A?pwd=2023 ),pwd:2023 😋😋😋

## Dataset Path Setting
```
 LEVIR-CD or WHU-CD  
     |—train  
          |   |—A  
          |   |—B  
          |   |—label  
     |—val  
          |   |—A  
          |   |—B  
          |   |—label  
     |—test  
          |   |—A  
          |   |—B  
          |   |—label
  ```        
 Where A contains images of the first temporal image, B contains images of the second temporal images, and the label contains ground truth maps.  
## Dataset mean and std setting 
We calculated mean and std for seven data sets in line 27-38 of `utils/datasetHCX` , you can use one directly and then annotate the others.
```bash
# It is for LEVIR!
# self.mean1, self.std1, self.mean2, self.std2 =[0.45025915, 0.44666713, 0.38134697],[0.21711577, 0.20401315, 0.18665968],[0.3455239, 0.33819652, 0.2888149],[0.157594, 0.15198614, 0.14440961]
# It is for WHU!
self.mean1, self.std1, self.mean2, self.std2 = [0.49069053, 0.44911194, 0.39301977], [0.17230505, 0.16819492,0.17020544],[0.49139765,0.49035382,0.46980983], [0.2150498, 0.20449342, 0.21956162]
```

## PFBS(Progressive Foreground-Balanced Sampling)
you can set `Normal Train`,`Fixed-X`,`Linear-Y`,`Fixed-X Linear-Y` method in line 113-135 of `trainHCX.py` .You just need to choose one sampling method, and annotate the others, About 'X' and 'Y', you can set `epochs_threshold` number in `metadata.json`.
![image-20230415](/picture/PFBS-2.png)
```bash
#Normal Train：正常训练，确保dataloader的方式一样
# train_loader.dataset.curr_num = len(train_loader.dataset)
#Fixed-X:如固定的15个
if epoch < opt.epochs_threshold:
   pass
else:  # 15
   train_loader.dataset.curr_num=len(train_loader.dataset)
#Fixed-X Linear-Y：先固定，后增加，前10个是前景影像，然后线性增加10个，后是正常训练
# if epoch < opt.epochs_threshold:
#     pass
# elif epoch<opt.epochs_threshold+5:
#     train_loader.dataset.curr_num += add_per_epoch
# else:  # 20
#     train_loader.dataset.curr_num=len(train_loader.dataset)
# # Linear-Y：前20个线性增加
# if epoch == 0:
#     pass
# elif epoch < opt.epochs_threshold:  # 20
#     train_loader.dataset.curr_num += add_per_epoch
# else:
#     train_loader.dataset.curr_num = len(train_loader.dataset)
```


![image-20230415](/picture/HANet-HCGMNet-CGNet.png)
## Citation 

 If you use this code for your research, please cite our papers.  

```
@ARTICLE{10093022,
  author={Han, Chengxi and Wu, Chen and Guo, Haonan and Hu, Meiqi and Chen, Hongruixuan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images}, 
  year={2023},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/JSTARS.2023.3264802}}


```
## Acknowledgments
 
 Our code is inspired and revised by [pytorch-MSPSNet](https://github.com/QingleGuo/MSPSNet-Change-Detection-TGRS),[pytorch-SNUNet](https://github.com/likyoo/Siam-NestedUNet), Thanks  for their great work!!  



## Reference  
[1] C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“[HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/10093022),” IEEE J. SEL. TOP. APPL.EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.


[2] [HCGMNET: A Hierarchical Change Guiding Map Network For Change Detection](https://doi.org/10.48550/arXiv.2302.10420).

[3]C. Wu et al., "[Traffic Density Reduction Caused by City Lockdowns Across the World During the COVID-19 Epidemic: From the View of High-Resolution Remote Sensing Imagery](https://ieeexplore.ieee.org/abstract/document/9427164)," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 5180-5193, 2021, doi: 10.1109/JSTARS.2021.3078611.


