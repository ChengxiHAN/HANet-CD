
# HANet-Change-Detection :https://chengxihan.github.io/
The Pytorch implementation for:
HANET: A HIERARCHICAL ATTENTION NETWORK FOR CHANGE DETECTION WITH BI-TEMPORAL VERY-HIGH-RESOLUTION REMOTE SENSING IMAGES,” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.
 C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
 https://ieeexplore.ieee.org/abstract/document/10093022

[14 Aril. 2023] Release the first version of the HANet

__Dataset Download__   
 LEVIR-CD：https://justchenhao.github.io/LEVIR/  
 
 WHU-CD：http://gpcv.whu.edu.cn/data/building_dataset.html


 Note: Please crop the LEVIR dataset to a slice of 256×256 before training with it.  

__Dataset Path Setteing__  
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
          
 Where A contains images of first temporal image, B contains images of second temporal images, and label contains groundtruth maps.  

__Traing and test Process__   

 python trainHCX.py  
 python test.py
 python Output_Results.py

__Revised parameters__  
 You can revised related parameters in the "metadata.json" file.  

__Requirement__  

-Pytorch 1.8.0  
-torchvision 0.9.0  
-python 3.8  
-opencv-python  4.5.3.56  
-tensorboardx 2.4  
-Cuda 11.3.1  
-Cudnn 11.3  


__Citation__  

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
__Acknowledgments__  

 Our code is inspired and revised by [pytorch-MSPSNet],[pytorch-SNUNet], Thanks  for their great work!!  

__Reference__  
[1] C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“HANET: A HIERARCHICAL ATTENTION NETWORK FOR CHANGE DETECTION WITH BI-TEMPORAL VERY-HIGH-RESOLUTION REMOTE SENSING IMAGES,” IEEE J. SEL. TOP. APPL.EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.
https://ieeexplore.ieee.org/abstract/document/10093022

[2] HCGMNET: A Hierarchical Change Guiding Map Network For Change Detection, https://doi.org/10.48550/arXiv.2302.10420

[3]C. Wu et al., "Traffic Density Reduction Caused by City Lockdowns Across the World During the COVID-19 Epidemic: From the View of High-Resolution Remote Sensing Imagery," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 5180-5193, 2021, doi: 10.1109/JSTARS.2021.3078611.
https://ieeexplore.ieee.org/abstract/document/9427164

