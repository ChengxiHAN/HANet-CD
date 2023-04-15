import numpy as np
from tqdm import tqdm
from skimage import io
from pandas import read_csv
import os

PATH_DATASET = '/data/chengxi.han/data/S2Looking/'
PATH_TRAIN_DATASET = PATH_DATASET + 'train/label'
PATH_VAL_DATASET = PATH_DATASET + 'val/label'
PATH_TEST_DATASET = PATH_DATASET + 'test/label'

print('影像为：',PATH_DATASET)
def get_weights(path):
    fname = os.listdir(path)

    n_pix = 0
    true_pix = 0

    for img_file_name in tqdm(fname, position=0, desc="读取影像"):
        cm = io.imread(os.path.join(path,img_file_name), as_gray=True) != 0

        s = cm.shape
        n_pix += np.prod(s)
        true_pix += cm.sum()

    return [n_pix, true_pix, n_pix-true_pix]

weights = get_weights(path=PATH_TRAIN_DATASET)
print("\ntrain总像素|变化的|未变化的：",weights)
weights = get_weights(path=PATH_VAL_DATASET)
print("\nval总像素|变化的|未变化的：",weights)
weights = get_weights(path=PATH_TEST_DATASET)
print("\ntest总像素|变化的|未变化的：",weights)
