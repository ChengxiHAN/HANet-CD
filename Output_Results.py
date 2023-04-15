'''
# Qingle Guo "Deep Multiscale Siamese Network with Parallel Convolutional Structure and Self-Attention for Change Detection" TGRS-2021
# showing the detection results of the test dataset (LEVIR-CD and SYSU)
'''

import torch.utils.data
import os
import cv2
from tqdm import tqdm
from utils.parser import get_parser_with_args
from utils.Related import get_test_loaders, initialize_metrics

from datasetHCX import HCXDataset
from torch.utils.data import DataLoader



# model setting, weighted setting and dataloder
parser, metadata = get_parser_with_args()
opt = parser.parse_args()
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="2"

# test_loader = get_test_loaders(opt, batch_size=16)
# testdataset = HCXDataset('/data/chengxi.han/data/Building change detection dataset256/test', is_training=False) #is_training=False即为取全部影像，不仅取前景
testdataset = HCXDataset(opt.dataset_dir +'test', is_training=False)
test_loader = DataLoader(testdataset, batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers) #batch size一定要是1！！！！


# path = 'tmp/WHU-ChangedLabelTrain-Fixed15/epoch_40.pt'   # the path of the model
path = opt.weight_dir+'final_epoch49.pt'
# path = opt.weight_dir+'epoch_39.pt'
# path = opt.weight_dir+'best.pt'
model = torch.load(path)

# if not os.path.exists('Detection_Re/WHU-ChangedLabelTrain-Fixed15'):
#     os.mkdir('Detection_Re/WHU-ChangedLabelTrain-Fixed15')

if not os.path.exists(opt.Output_dir):
    os.mkdir(opt.Output_dir)

# test processing
model.eval()
Img_index = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    # Unpacking
    T = tqdm(test_loader)
    for Imgs1, Imgs2, labels in T:
        # Transferring to the device
        Imgs1 = Imgs1.float().to(dev)
        Imgs2 = Imgs2.float().to(dev)
        # labels = labels.long().to(dev)

        # Model output
        Output = model(Imgs1, Imgs2)
        # print('输出影像的元组大小为：', len(Output))
        Output = Output[-1]
        # print('输出影像的元组大小为：', len(Output))
        _, Output = torch.max(Output, 1)
        # print('输出影像的元组大小为：', len(Output))
        Output = Output.data.cpu().numpy()
        Output = Output.squeeze() * 255
        # print('输出影像的一维度大小为：',Output.shape[0],'输出影像的二维度大小为：',Output.shape[1],'输出影像的三维度大小为：',Output.shape[2])
        # print('输出影像的一维度大小为：',Output.shape[0],'输出影像的二维度大小为：',Output.shape[1])
        # Output=Output[0:1,:,:]
        # print('输出影像的一维度大小为：', Output.shape[0], '输出影像的二维度大小为：', Output.shape[1], '输出影像的三维度大小为：', Output.shape[2])
        # results saving
        file_path = 'Detection_Re/S2Looking-Fixed15-Row-ColAttention-Epo50/' + str(Img_index).zfill(1)
        # file_path = opt.Output_dir + str(Img_index).zfill(1)

        # cv2.imshow('结果',Output)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(file_path + '.png', Output)
        cv2.imwrite(opt.Output_dir + testdataset.ids[Img_index], Output)


        Img_index += 1

if __name__ == '__main__':
    print('测试影像')