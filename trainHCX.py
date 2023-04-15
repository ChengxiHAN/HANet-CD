"""
training and validation process
Please cited:
HANet (Chengxi Han韩承熙,Wuhan University,chengxihan@whu.edu.cn,https://chengxihan.github.io/)
please Cited the paper:
C. HAN, C. WU, H. GUO, M. HU, AND H. CHEN, 
“HANET: A HIERARCHICAL ATTENTION NETWORK FOR CHANGE DETECTION WITH BI-TEMPORAL VERY-HIGH-RESOLUTION REMOTE SENSING IMAGES,” IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3264802.

we attempt to illustrate the whole process to make the reader understood.
"""

import datetime
import torch
import os
import logging
import random
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.Related import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics,get_loaders2)
from tensorboardX import SummaryWriter
from torchsummary import summary
import numpy as np
import cv2
from utils.datasetHCX import HCXDataset
from utils.losses import hybrid_loss,FocalLoss
from torch.utils.data import DataLoader
import os
from torch.cuda.amp import autocast as autocast
# from torch.cuda.amp import Gradscaler
import time
torch.backends.cudnn.enabled = False

time_start = time.perf_counter()  # 记录开始时间
print('现在的时间为：',time_start)

warnings.filterwarnings("ignore")

"""Initialize Parser and define arguments"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""Initialize experiments log"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')
print('\n现在的时间是：',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),'\n')

"""Set up environment: define paths, download data, and set device"""
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))
# ensuring the same value after shuffle
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch(seed=777)



if __name__ == '__main__':

    traindataset = HCXDataset(opt.dataset_dir + 'train', is_training=True)
    train_loader = DataLoader(traindataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
     valdataset = HCXDataset(opt.dataset_dir + 'val', is_training=False)
    val_loader = DataLoader(valdataset, batch_size=opt.batch_size, shuffle=False, drop_last=False,
                            num_workers=opt.num_workers)

    logging.info('Model LOADING')
    model = load_model(opt, dev)


    criterion = hybrid_loss(gamma=opt.gamma)#get_criterion(opt) 之前用的这个
    # criterion =FocalLoss(gamma=opt.gamma) #现在想用这个FocalLoss

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate,weight_decay=opt.weight_decay) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5) #每过8个epoch更新一次
    scaler=torch.cuda.amp.GradScaler() #训练前实例化一个GradScaler对象
    best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
    logging.info('Training Start')
    total_step = -1
    """ Training process"""
    print('Labeled前景影像数量:', train_loader.dataset.labeled_num )
    print('Total训练影像总数量:', len(train_loader.dataset) )
    print('Epochs_Threshold:',opt.epochs_threshold)
    add_per_epoch=(len(train_loader.dataset) -  train_loader.dataset.labeled_num) //opt.epochs_threshold #分母为增加的数量，这里为固定的的训练
    # add_per_epoch = (len(train_loader.dataset) - train_loader.dataset.labeled_num) // 10 #这里为先固定后增加的数量，如先固定5再增加10
    print('每个Epoch增加',add_per_epoch,'个')
    for epoch in range(opt.epochs):
        train_metrics = initialize_metrics()
        val_metrics = initialize_metrics()
        model.train()

        logging.info('MSPSNet model training!!!')

        print('\n', '现在训练的是"HANet"网络', '\n')
        print('此次训练的Epoch=', opt.epochs, '\n')
        print('此次训练的Batch_Size =', opt.batch_size, '\n')
        print('此次训练的路径 =c', opt.dataset_dir, '\n')
        print('权重保存的路径 =', opt.weight_dir, '\n')
        print('阈值 =', opt.epochs_threshold, '\n')
        # print('loss中gamma =',gamma, '\n')

#----------------------这里修改PFBS的方式，只采用一种即可，其他注释掉-------------------------------------
        #正常训练，确保dataloader的方式一样
        # train_loader.dataset.curr_num = len(train_loader.dataset)
        #固定的15个
        if epoch < opt.epochs_threshold:
            pass
        else:  # 20
            train_loader.dataset.curr_num=len(train_loader.dataset)
        #先固定，后增加，前10个是前景影像，然后线性增加10个，后是正常训练
        # if epoch < opt.epochs_threshold:
        #     pass
        # elif epoch<opt.epochs_threshold+5:
        #     train_loader.dataset.curr_num += add_per_epoch
        # else:  # 20
        #     train_loader.dataset.curr_num=len(train_loader.dataset)
        # # 前20个线性增加
        # if epoch == 0:
        #     pass
        # elif epoch < opt.epochs_threshold:  # 20
        #     train_loader.dataset.curr_num += add_per_epoch
        # else:
        #     train_loader.dataset.curr_num = len(train_loader.dataset)
#----------------------这里修改PFBS的方式，只采用一种即可，其他注释掉-------------------------------------

        batch_iter = 0
        tbar = tqdm(train_loader)
        temp_label_num = []  # 统计影像数量
        # load training dataset
        for batch_img1, batch_img2, labels in tbar:
            start_time=time.time()
            tbar.set_description(
                "epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter + opt.batch_size))
            print('Epoch',epoch,(labels.sum(-1).sum(-1) > 0).sum(), '个')
            batch_iter = batch_iter + opt.batch_size
            total_step += 1


            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)
            optimizer.zero_grad()

 
            cd_preds = model(batch_img1, batch_img2)
            cd_loss = criterion(cd_preds, labels)
            loss = cd_loss
            loss.backward()
            optimizer.step()

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size ** 2)))

            cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                                   cd_preds.data.cpu().numpy().flatten(),
                                   average='binary',
                                   pos_label=1, zero_division=1)

            train_metrics = set_metrics(train_metrics,
                                        cd_loss,
                                        cd_corrects,
                                        cd_train_report,
                                        scheduler.get_last_lr())

            mean_train_metrics = get_mean_metrics(train_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'train': v}, total_step)

            del batch_img1, batch_img2, labels


        scheduler.step()
        logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

        """Validation process"""
        model.eval()
        with torch.no_grad():
            for batch_img1, batch_img2, labels in val_loader:
                # Set variables for training
                batch_img1 = batch_img1.float().to(dev)
                batch_img2 = batch_img2.float().to(dev)
                labels = labels.long().to(dev)

                cd_preds = model(batch_img1, batch_img2)

                cd_loss = criterion(cd_preds, labels)

                cd_preds = cd_preds[-1]
                _, cd_preds = torch.max(cd_preds, 1)

                cd_corrects = (100 *
                               (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                               (labels.size()[0] * (opt.patch_size ** 2)))

                cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                     cd_preds.data.cpu().numpy().flatten(),
                                     average='binary',
                                     pos_label=1, zero_division=1)

                val_metrics = set_metrics(val_metrics,
                                          cd_loss,
                                          cd_corrects,
                                          cd_val_report,
                                          scheduler.get_last_lr())

                # log the batch mean metrics
                mean_val_metrics = get_mean_metrics(val_metrics)

                for k, v in mean_train_metrics.items():
                    writer.add_scalars(str(k), {'val': v}, total_step)

                # clear batch variables from memory
                del batch_img1, batch_img2, labels

            logging.info("EPOCH {} VALIDATION METRICS".format(epoch) + str(mean_val_metrics))

            """
            Store the weights of good epochs based on validation results
            """

            if ( mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores']):

                # Insert training and epoch information to metadata dictionary
                logging.info('updata the model')
                metadata['validation_metrics'] = mean_val_metrics


                if not os.path.exists(opt.weight_dir):
                    os.mkdir(opt.weight_dir)
                torch.save(model, opt.weight_dir+'epoch_' + str(epoch) + '.pt')
                torch.save(model, opt.weight_dir + 'best' + '.pt')   #

                # comet.log_asset(upload_metadata_file_path)
                best_metrics = mean_val_metrics

            print('An epoch finished.')

    torch.save(model, opt.weight_dir + 'final_epoch' + str(epoch) + '.pt') #方便
    writer.close()  # close tensor board
    print('Done!')

time_end = time.perf_counter()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('总时间为：h',time_sum/3600)