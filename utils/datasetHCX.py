"""
HANet (Chengxi Han韩承熙,Wuhan University,chengxihan@whu.edu.cn,https://chengxihan.github.io/)
please Cited the paper:
C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“HANET: A HIERARCHICAL ATTENTION NETWORK FOR CHANGE DETECTION WITH BI-TEMPORAL VERY-HIGH-RESOLUTION REMOTE SENSING IMAGES,” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.
"""

import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

from PIL import Image

import torchvision.transforms.functional as transF

from tqdm import tqdm
from imgaug import augmenters as iaa
import random
import os
import cv2

class HCXDataset(Dataset):
    def __init__(self, data_dir, is_training=True):
        self.data_dir = data_dir
        self.is_training=is_training
        # It is for CDD!
        # self.mean1, self.std1, self.mean2, self.std2 =[0.35390204, 0.3910402, 0.3430754],[0.21585055, 0.23398526, 0.2087468], [0.47324985, 0.49860582, 0.46874452],[0.24311192, 0.2601877, 0.25678152]
        # It is for LEVIR!
        # self.mean1, self.std1, self.mean2, self.std2 =[0.45025915, 0.44666713, 0.38134697],[0.21711577, 0.20401315, 0.18665968],[0.3455239, 0.33819652, 0.2888149],[0.157594, 0.15198614, 0.14440961]
        # It is for WHU!
        self.mean1, self.std1, self.mean2, self.std2 = [0.49069053, 0.44911194, 0.39301977], [0.17230505, 0.16819492,0.17020544], [0.49139765,0.49035382,0.46980983], [0.2150498, 0.20449342, 0.21956162]
        # It is for DSIFN!
        # self.mean1, self.std1, self.mean2, self.std2 =[0.39334667, 0.41008812, 0.372291],[0.20001005, 0.19594198, 0.19641198],[0.39037502, 0.3820759, 0.38505003],[0.19432788, 0.17783593, 0.17126249]
        # It is for SYSU!
        # self.mean1, self.std1, self.mean2, self.std2 =[0.39659828, 0.5284582, 0.46539742],[0.24339822, 0.1843563, 0.18066375],[0.40202096, 0.48765418, 0.39895305],[0.21836075, 0.18112999, 0.17903014]
        # It is for S2Looking!
        # self.mean1, self.std1, self.mean2, self.std2 = [0.17553517, 0.20817895, 0.20358624],[0.07853421, 0.080604345, 0.07777272],[0.20552409, 0.21961312, 0.21183391],[0.07319275, 0.06418888, 0.06009885]


        # self.ids = [file for file in os.listdir(os.path.join(data_dir,'A'))] #获取总数据集
        #-------------------------------------------获取一定比例的数据集
        Total_ids = [file for file in os.listdir(os.path.join(data_dir,'A'))]
        TraingingsetRate =1 #0.05 #训练数据集的影像比例
        numberOfTrain = int(len(Total_ids)*TraingingsetRate)
        self.ids = random.sample(Total_ids, numberOfTrain)
        # -------------------------------------------获取一定比例的数据集

        if self.is_training:
            label_stat=[]
            for id in tqdm(self.ids):
                label=np.array(Image.open(os.path.join(self.data_dir, 'label', id)))
                label_stat.append((label>0).sum())
            label_stat=np.array(label_stat)
            self.index=np.argsort(np.array(label_stat))[::-1]
            self.labeled_num=(label_stat>0).sum()
            self.curr_num=(label_stat>0).sum()
            print('Training Stage:')
            print('Loaded',self.index.shape[0], 'Images')
            print('Include', (label_stat>0).sum(), 'Labeled')
            self.transform = iaa.Sequential([
                iaa.Rot90([0,1,2,3]),
                iaa.VerticalFlip(p=0.5),
                iaa.HorizontalFlip(p=0.5)
            ])
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        # if self.is_training:
        #     i=i%self.curr_num
        #     i=self.index[i]
        if self.is_training:
            idx = i % self.curr_num
            # i = self.index[idx]
            i = self.index[idx] if idx < self.labeled_num else self.index[i]
        idx = self.ids[i]
        mask_file =os.path.join(self.data_dir,'label', idx )
        imgA_file = os.path.join(self.data_dir,'A', idx )
        imgB_file = os.path.join(self.data_dir, 'B', idx)
        mask = np.array(Image.open(mask_file))
        imgA = np.array(Image.open(imgA_file))
        imgB = np.array(Image.open(imgB_file))

        if len(mask.shape)==3:
            mask=mask[:,:,0]
        if self.is_training:
            img,mask=self.transform(image=np.concatenate([imgA,imgB],-1),segmentation_maps=mask[np.newaxis,:,:,np.newaxis])
            imgA,imgB=img[:,:,:3],img[:,:,3:]
            mask=mask[0,:,:,0]

        imgA,imgB,mask=transF.to_tensor(imgA.copy()),transF.to_tensor(imgB.copy()),(transF.to_tensor(mask.copy())>0).int()
        imgA,imgB=transF.normalize(imgA,self.mean1,self.std1),transF.normalize(imgB,self.mean2,self.std2)

        #image_data = np.asarray(imgA)
        #file_path = '/Detection_Re/dataloaderTest/' + idx
        # cv2.imwrite(file_path, image_data)

        return imgA.float(), imgB.float(), mask.float()



        # return {
        #     'imgA': imgA.float(),
        #     'imgB': imgB.float(),
        #     'mask': mask.float(),
        #     'name':idx
        # }
if __name__ == '__main__':
    import datetime
    import os
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from torch.utils.data import DataLoader
    traindataset = HCXDataset('/data/chengxihan/LEVIR256/train',is_training=True)
    train_loader = DataLoader(traindataset, batch_size=16, shuffle=False, num_workers=24)
    num_epochs=5
    for epoch in range(num_epochs):
        if epoch<5:
            curr_num=train_loader.dataset.curr_num
            labeled_num=train_loader.dataset.labeled_num
            train_loader.dataset.curr_num=curr_num+(len(train_loader.dataset)-labeled_num)//5*epoch
        elif epoch>=5:
            train_loader.dataset.curr_num=len(train_loader.dataset)
        for num, batch in enumerate(train_loader):
            imgsA, imgsB = batch['imgA'], batch['imgB']
            true_masks = batch['mask'] > 0
            test=true_masks.sum(-1).sum(-1).sum(-1)>0
            print(test)
            # assert test.all()==True
