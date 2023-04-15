
# HANet-Change-Detection :https://chengxihan.github.io/
The Pytorch implementation for::gift::gift::gift:
“[HANet: A hierarchical attention network for change detection with bi-temporal very-high-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/10093022),” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.
 C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, :yum::yum::yum:

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
 You can revised related parameters in the `metadata.json` file.  
 
## Traing,Test and Visualization Process   

```bash
python trainHCX.py 
python test.py 
python Output_Results.py
```

## Dataset Download   
 LEVIR-CD：https://justchenhao.github.io/LEVIR/  
 
 WHU-CD：http://gpcv.whu.edu.cn/data/building_dataset.html

 Note: Please crop the LEVIR dataset to a slice of 256×256 before training with it. 
 ![image-20230415](/picture/HANet-ExperimentResult.png)

## Dataset Path Setting
```
 LEVIR-CD or WHU-CD  
     |—train  
          |   |—A  
          |   |—B  
          |   |—lable  
     |—val  
          |   |—A  
          |   |—B  
          |   |—lable  
     |—test  
          |   |—A  
          |   |—B  
          |   |—lable
  ```        
 Where A contains images of first temporal image, B contains images of second temporal images, and label contains groundtruth maps.  
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


