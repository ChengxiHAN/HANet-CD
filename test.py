"""
Please cited:
HANet (Chengxi Han韩承熙,Wuhan University,chengxihan@whu.edu.cn,https://chengxihan.github.io/)
please Cited the paper:
C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“HANET: A HIERARCHICAL ATTENTION NETWORK FOR CHANGE DETECTION WITH BI-TEMPORAL VERY-HIGH-RESOLUTION REMOTE SENSING IMAGES,” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.
"""
import warnings
import torch.utils.data
from tqdm import tqdm
from utils.parser import get_parser_with_args
from utils.Related import get_test_loaders
from sklearn.metrics import confusion_matrix
from utils.datasetHCX import HCXDataset
from torch.utils.data import DataLoader
import os

warnings.filterwarnings("ignore")

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# test_loader = get_test_loaders(opt)
# testdataset = HCXDataset('/data/chengxi.han/data/Building change detection dataset256/test', is_training=False) #is_training=False即为取全部影像，不仅取前景
testdataset = HCXDataset(opt.dataset_dir +'test', is_training=False)
test_loader = DataLoader(testdataset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
                        num_workers=opt.num_workers)

"""load the weighted file and model"""
# path = 'tmp/WHU-ChangedLabelTrain-Fixed15-LinearIncrease10/best.pt'
path = opt.weight_dir+'final_epoch99.pt'
# path = opt.weight_dir+'epoch_46.pt'

model = torch.load(path)
print('现在的路径是：',path)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        cd_preds.data.cpu().numpy().flatten(),labels=[0,1]).ravel()

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
# P = tp / (tp + fp)
# R = tp / (tp + fn)
# F1 = 2 * P * R / (R + P)
#
# print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))
print('tp:',tp)
print('tn',tn)

FA= fp / (tp+fn)
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)
OA = (tp+tn)/(tp+fp+tn+fn)
PRE=((tp+fp)*(tp+fn) + (tn+fn)*(fp+tn)) / ((tp+fp+tn+fn)*(tp+fp+tn+fn))
Kappa= (OA-PRE)/(1-PRE)
IoU=tp/(tp+fn+fp)
print('F1-Score: {:.2f}\nPrecision: {:.2f}\nRecall: {:.2f}\nOA: {:.2f}\nKappa: {:.2f}\nIoU: {:.2f}\nFA: {:.2f}'.format(F1*100,P*100, R*100, OA*100,Kappa*100,IoU*100,FA*100))
print('{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}'.format(F1*100,P*100, R*100, OA*100,Kappa*100,IoU*100))
print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(F1*100,P*100, R*100, OA*100,Kappa*100,IoU*100))

MPA=0.5*(tp/(tp+fn)+tn/(tn+fp))
FWIoU=(1/(tp+fp+tn+fn))*(tp*(tp+fn)/(fp+fn+tp)+tn*(tn+fp)/(fp+fn+tn))
Dice=2*tp/(fp+2*tp+fn)
cIoU=IoU
ucIoU=tn/(tn+fp+fn)
mIoU=(cIoU+ucIoU)/2
print('MPA: {:.2f}\nFWIoU: {:.2f}\nDice: {:.2f}\ncIoU: {:.2f}\nucIoU: {:.2f}\nmIoU: {:.2f}'.format(MPA*100,FWIoU*100, Dice*100, cIoU*100,ucIoU*100,mIoU*100))
print('{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}'.format(MPA*100,FWIoU*100, Dice*100, cIoU*100,ucIoU*100,mIoU*100))
print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(MPA*100,FWIoU*100, Dice*100, cIoU*100,ucIoU*100,mIoU*100))