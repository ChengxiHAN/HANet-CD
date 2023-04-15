import numpy as np
import cv2
import os

img_h, img_w = 256, 256  # 根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []
imgs_path = '/data/chengxi.han/data/S2Looking256/train/B'  # 路径自己修改
imgs_path_list = os.listdir(imgs_path)  # imgs

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path, item), -1)
    img = cv2.resize(img, (img_w, img_h))
    img = np.reshape(img, (256, 256, -1))
    img = img[:, :, :, np.newaxis]  # 前三个：代表的是遍历行，列，通道数，最后np.newaxis新增第四维度
    img_list.append(img)
    i += 1
    print(i, '/', len_)

imgs = np.concatenate(img_list, axis=3)  # axis=3代表按照第四维度拼接起来
imgs = imgs.astype(np.float32) / 255.

for i in range(3):  # 如果是rgb图的话，改为range(3)
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
# temp_label_num.append((labels > 0).sum())
# print('\n','batch中label有1的数量为', temp_label_num, '个')

