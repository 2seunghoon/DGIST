from __future__ import print_function
from json.encoder import py_encode_basestring_ascii
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
from tkinter import Image
import traceback
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.backends.cudnn
import numpy as np
import torch.nn.functional as F
import cv2

# from Networks import *
from deeplabv2 import Deeplab
from miou import *

from utils import *
from losses import *
from PIL import Image
from dataloader.Cityscapes import decode_labels
from dataset import get_dataset
from param import get_params
from prettytable import PrettyTable
from losses import Losses
from sklearn.manifold import TSNE
from things_stuff_transfer import *
import matplotlib.pyplot as plt
import pandas as pd

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.imsize = opt.imsize
        self.best_miou = 0

        self.train_loader, self.test_loader, self.data_iter = dict(), dict(), dict()
        for dset in self.opt.datasets:
            train_loader, test_loader = get_dataset(dataset=dset, batch=self.opt.batch,
                                                    imsize=self.imsize, workers=self.opt.workers, super_class=opt.super_class)
            self.train_loader[dset] = train_loader
            self.test_loader[dset] = test_loader
        self.centroid_list=list()
        self.label_list=list()
        self.nets, self.optims, self.losses = dict(), dict(), dict()
        self.writer = SummaryWriter('./tensorboard/%s' % opt.ex)
        self.logger = getLogger()
        self.step = 0
        self.checkpoint = './checkpoint/%s' % opt.ex
        self.source = opt.datasets[0] # G 
        self.target = opt.datasets[1] # C
        self.targets = opt.datasets[1:]
        self.loss_fns = Losses(self.opt)
        self.n_class = 19
        self.name_classes = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "trafflight", "traffsign", "vegetation",
            "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
        ]
        self.ImageNet_classes = [
            6,7,11,12,13,14,15,16,17,18
        ]
        self.x1_centroid_list = torch.empty((0,2048))
        self.x1_centroid_list_label = torch.empty((0))
        self.global_i = 0

    def set_default(self):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False

        ## Random Seed ##
        print("Random Seed: ", self.opt.manualSeed)
        seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        torch.cuda.manual_seed_all(self.opt.manualSeed)

        ## Logger ##
        file_log_handler = FileHandler(self.opt.logfile)
        self.logger.addHandler(file_log_handler)
        stderr_log_handler = StreamHandler(sys.stdout)
        self.logger.addHandler(stderr_log_handler)
        self.logger.setLevel('INFO')
        formatter = Formatter()
        file_log_handler.setFormatter(formatter)
        stderr_log_handler.setFormatter(formatter)

    def save_networks(self):
        miou = ''
        miou = miou + '%.2f'%self.best_miou
        if not os.path.exists(self.checkpoint+'/%s' % miou):
            os.mkdir(self.checkpoint+'/%s' % miou)
        for key in self.nets.keys():
            torch.save(self.nets[key].state_dict(), self.checkpoint + '/%s/net%s.pth' % (miou, key))

    def set_networks(self):
        # self.nets['T'] = Deeplab(num_classes=self.n_class, init_weights='pretrained/deeplab_gta5.pth')
        self.nets['T'] = Deeplab(num_classes=19, restore_from='pretrained/deeplab_gta5')
        # self.nets['ImageNet'] = Deeplab(num_classes=self.n_class, init_weights='pretrained/deeplab_gta5.pth')

        for net in self.nets.keys():
            self.nets[net].cuda()

    def  set_optimizers(self):
        self.optims['T'] = optim.SGD(self.nets['T'].parameters(), lr=self.opt.lr_seg, momentum=0.9,
                                         weight_decay=self.opt.weight_decay_task)

    def set_zero_grad(self):
        for net in self.nets.keys():
            self.nets[net].zero_grad()

    def set_train(self):
        for net in self.nets.keys():
            self.nets[net].train()

    def set_eval(self):
        self.nets['T'].eval()

    def get_batch(self, batch_data_iter):

        batch_data = dict()
        for dset in self.opt.datasets:
            try:
                batch_data[dset] = batch_data_iter[dset].next()
            except StopIteration:
                batch_data_iter[dset] = iter(self.train_loader[dset])
                batch_data[dset] = batch_data_iter[dset].next()
        return batch_data

    def plot_centroid(self, cat_features, cat_labels):
        object_col = []
        
        for i in range(len(cat_features)):
            if cat_features[i].sum() != 0:
                object_col.append(i)
        cat_features = cat_features[object_col]
        cat_labels = cat_labels[object_col]

        if self.x1_centroid_list.size == 0:
            self.x1_centroid_list = cat_features.detach().cpu()
            self.x1_centroid_list_label = cat_labels.detach().cpu()
        else:
            self.x1_centroid_list = torch.cat((self.x1_centroid_list, cat_features.detach().cpu()),dim=0)
            self.x1_centroid_list_label = torch.cat((self.x1_centroid_list_label, cat_labels.detach().cpu()), dim=0)
        self.global_i += 1
        if self.global_i % 50 == 0:
            # 2차원으로 차원 축소
            n_components = 2
            # t-sne 모델 생성
            model = TSNE(n_components=n_components)

            tsne_np = model.fit_transform(self.x1_centroid_list)
            # numpy array -> DataFrame 변환
            tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
            tsne_df['class_num'] = self.x1_centroid_list_label
            fig, ax = plt.subplots()
            scatter = ax.scatter(tsne_df['component 0'], tsne_df['component 1'], c=tsne_df['class_num'], label=tsne_df['class_num'])
            legend1 = ax.legend(*scatter.legend_elements())
            ax.add_artist(legend1)
            plt.xlabel('component 0')
            plt.ylabel('component 1')
            # plt.legend(tsne_df.index.tolist(), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])
            plt.savefig(f'./checkpoint/{self.opt.ex}/centroid_class_{self.global_i}.png')
            # self.x1_centroid_list = torch.empty((0,2048))
            # self.x1_centroid_list_label = torch.empty((0))

    def train_task(self, imgs, labels,coco_image,coco_gt):  # Train Task Networks (T)
        # self.set_zero_grad()
        *_, GTA_feature = self.nets['T'](imgs[self.source], lbl=labels[self.source])
        *_, coco_feature = self.nets['T'](coco_image[self.source], lbl=coco_gt[self.source])

        gta_centroid = self.loss_fns.get_centroid_gta(GTA_feature,labels[self.source], self.ImageNet_classes).detach().cpu().numpy()
        coco_centroid = self.loss_fns.get_centroid_gta(coco_feature,coco_gt[self.source], self.ImageNet_classes).detach().cpu().numpy()  # 19,2048

        # gta_centroid = self.loss_fns.get_centroid_gta(GTA_feature,labels[self.source], self.ImageNet_classes)
        # coco_centroid = self.loss_fns.get_centroid_gta(coco_feature,coco_gt[self.source], self.ImageNet_classes)  # 19,2048
        # cat_features = torch.cat((gta_centroid, coco_centroid), dim=0)
        # cat_labels = torch.empty((0), device="cuda")
        # import pdb; pdb.set_trace()
        
        # for i in range(len(gta_centroid)):
        #     num = torch.tensor([i], device="cuda")
        #     cat_labels = torch.cat((cat_labels, num), dim=0)

        # for i in range(len(coco_centroid)):
        #     num = torch.tensor([i], device="cuda")
        #     cat_labels = torch.cat((cat_labels, num), dim=0)
        # self.plot_centroid(cat_features, cat_labels)
        index_gta=[]
        index_coco=[]
        labels_gta=np.zeros(len(range(19))).reshape(-1,1)
        labels_coco=np.ones(len(range(19))).reshape(-1,1)
        same_label=[6,11,13,14,15,16,17,18]
        for i in range(19):
            if gta_centroid[i].max()==0 or not(i in same_label):
                index_gta.append(i)
        for i in range(19):
            if coco_centroid[i].max()==0:
                index_coco.append(i)

        gta_centroid=np.delete(gta_centroid,index_gta,axis=0)
        coco_centroid=np.delete(coco_centroid,index_coco,axis=0)
        labels_gta=np.delete(labels_gta,index_gta,axis=0)
        labels_coco=np.delete(labels_coco,index_coco,axis=0)
        # import pdb;pdb.set_trace()
        gta_centroid=torch.FloatTensor(gta_centroid)
        coco_centroid=torch.FloatTensor(coco_centroid)
        labels_gta=torch.LongTensor(labels_gta)
        labels_coco=torch.LongTensor(labels_coco)
        if self.x1_centroid_list.shape == (0,2048):
            self.x1_centroid_list = torch.cat((gta_centroid,coco_centroid),dim=0)
            self.x1_centroid_list_label = torch.cat((labels_gta,labels_coco),dim=0)
        else:
            self.x1_centroid_list = torch.cat((self.x1_centroid_list, torch.cat((gta_centroid,coco_centroid),dim=0)),dim=0)
            self.x1_centroid_list_label = torch.cat((self.x1_centroid_list_label, torch.cat((labels_gta,labels_coco),dim=0)), dim=0)
            #
        # self.centroid_list.append(np.vstack([gta_centroid,coco_centroid]))
        # self.label_list.append(np.vstack([labels_gta,labels_coco]))
        if self.step%50==0:
            # centroid=np.array(self.centroid_list).reshape(-1,2048)
            # label=np.array(self.label_list).reshape(-1,1)
            model=TSNE(n_components=2)
            tsne_np=model.fit_transform(self.x1_centroid_list)
            tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
            tsne_df['class_num'] = self.x1_centroid_list_label
            fig, ax = plt.subplots()
            # colormap={6:'black',11:'crimson',13:'dodgerblue',14:'lime',15:'plum',16:'orange',17:'slategray',18:'aqua'}
            domain_map={0:"red",1:"blue"}

            classes=['traffic light','person','car','truck','bus','train','motorcycle','bicycle']
            scatter = ax.scatter(tsne_df['component 0'], tsne_df['component 1'], c=tsne_df['class_num'].map(domain_map), label=tsne_df['class_num'])
            legend1 = ax.legend(*scatter.legend_elements())
            # ax.legend()
            ax.add_artist(legend1)
            plt.xlabel('component 0')
            plt.ylabel('component 1')
            # plt.legend(tsne_df.index.tolist(), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'])
            plt.savefig(f'./checkpoint/{self.opt.ex}/GTA_transfer_{self.step}.png')
            if self.step%500==0:
                self.x1_centroid_list = torch.empty((0,2048))
                self.x1_centroid_list_label = torch.empty((0))

    def tensor_board_log(self, imgs, labels,coco_image,coco_gt):
        if type(coco_image)== torch.Tensor:
            nrow = coco_image.shape[0]+2
        else:
            nrow=2
        ##
        ##
        # import pdb;pdb.set_trace()
        # Input Images & Recon Images
        # coco_gt=coco_gt.unsqueeze(1)
        for dset in self.opt.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/%s' % dset, x, self.step)
            # if dset=='G':
                # x = vutils.make_grid(coco_image[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
                # self.writer.add_image('1_Input_Images/%s' % 'COCO', x, self.step)
        if type(coco_image)== torch.Tensor:
            x = vutils.make_grid(coco_image.detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/%s' % 'COCO', x, self.step)
        ## transfer
        # x = vutils.make_grid(transfer['G'].detach(), normalize=True, scale_each=True, nrow=nrow)
        # self.writer.add_image('1_Input_Images/%s' % 'Transfer', x, self.step)
        # x = vutils.make_grid(imgnet_1['G'], normalize=True, scale_each=True, nrow=nrow)
        # self.writer.add_image('1_Input_Images/%s' % 'Stuff', x, self.step)
        # x = vutils.make_grid(imgnet_2['G'], normalize=True, scale_each=True, nrow=nrow)
        # self.writer.add_image('1_Input_Images/%s' % 'Things', x, self.step)
        ##     ##
        # Segmentation GT, Pred
        self.set_eval()
        preds = dict()
        coco_preds=dict()
        for dset in self.opt.datasets:
            x = decode_labels(labels[dset].detach(), num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('4_GT/%s' % dset, x, self.step)
            preds[dset] = self.nets['T'](imgs[dset])[0]
            if type(coco_image)== torch.Tensor:
                x = decode_labels(coco_gt.detach(), num_images=coco_gt.shape[0])
                x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
                self.writer.add_image('4_GT/%s' % 'COCO', x, self.step)
                coco_preds[dset]=self.nets['T'](coco_image[0].unsqueeze(0))[0]
        for key in preds.keys():
            pred = preds[key].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            x = decode_labels(pred, num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('5_Prediction/%s' % key, x, self.step)
        
        ## for transfer
        if type(coco_image)== torch.Tensor:
            key='G'
            trans_pred = coco_preds[key].data.cpu().numpy()
            trans_pred = np.argmax(trans_pred, axis=1)
            x = decode_labels(trans_pred, num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('5_Prediction/%s' % 'COCO', x, self.step)        
        #####


        for loss in self.losses.keys():
            self.writer.add_scalar('Losses/%s' % loss, self.losses[loss], self.step)
        self.set_train()
        x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('1_Input_Images/%s' % dset, x, self.step)
        
        # x = vutils.make_grid(imgs[self.opt.datasets[0]].detach(), normalize=True, scale_each=True, nrow=nrow)
        # self.writer.add_image('1_Input_Images/%s' % self.opt.datasets[0], x, self.step)


    def eval(self, target):
        self.set_eval()

        miou = 0.
        min_miou = 100.
        confusion_matrix = np.zeros((self.n_class,) * 2)
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(self.test_loader[target]):
                imgs, labels = imgs.cuda(), labels.cuda()
                labels = labels.long()
                pred, *_ = self.nets['T'](imgs)
                pred = pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                gt = labels.data.cpu().numpy()
                confusion_matrix += MIOU(gt, pred, num_class=self.n_class)
                score = np.diag(confusion_matrix) / (
                            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(
                        confusion_matrix))
                miou = 100 * np.nanmean(score)
               
                    
                progress_bar(batch_idx, len(self.test_loader[target]), 'mIoU: %.3f' % miou)
            score = 100 * np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix))
            score = np.round(score, 1)
            table = PrettyTable()
            table.field_names = self.name_classes
            table.add_row(score)
            # Save checkpoint.
            self.logger.info('======================================================')
            self.logger.info('Step: %d | mIoU: %.3f%%'
                            % (self.step, miou))
            self.logger.info(table)
            self.logger.info('======================================================')
            if miou > self.best_miou:
                self.save_networks()
                self.best_miou = miou
            
        self.set_train()

    def print_loss(self):
        self.logger.info(
            '[%d/%d] Origin: %.2f|cocoloss : %.2f|coralloss : %.2f| %.2f %s'
            % (self.step, self.opt.iter,
                self.losses['T'],self.losses['coco_seg_loss'], self.losses['coral_loss'],self.best_miou, self.opt.ex))

    def train(self):
        self.set_default()
        self.set_networks()
        # self.set_optimizers()
        self.set_train()
        # freeze
        for param in self.nets['T'].parameters():
            param.requires_grad = False
        self.set_eval()
        # #
        batch_data_iter = dict()
        
        for dset in self.opt.datasets:
            batch_data_iter[dset] = iter(self.train_loader[dset])

        for i in range(self.opt.iter):
            self.step += 1
            # get batch data

            batch_data = self.get_batch(batch_data_iter)
            imgs, labels ,transfer= dict(), dict(),dict()
            imgnet_1=dict()
            imgnet_2=dict()

            gtaimage=dict()

            forconcat=np.array(list())
            coco_image=dict()
            coco_gt=dict()
            for dset in self.opt.datasets:
                if dset=='G':
                    # imgs[dset], labels[dset],imgnet[dset]= batch_data[dset]
                    imgs[dset], labels[dset],coco_image[dset],coco_gt[dset]= batch_data[dset]
                    imgs[dset], labels[dset],coco_image[dset],coco_gt[dset] = imgs[dset].cuda(), labels[dset].cuda(),coco_image[dset].cuda(),coco_gt[dset].cuda()
                    labels[dset] = labels[dset].long()
                    coco_gt[dset]=coco_gt[dset].long()
                elif dset=='C':
                    imgs[dset], labels[dset]= batch_data[dset]
                    imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                    labels[dset] = labels[dset].long()                   
            # training

            self.train_task(imgs, labels,coco_image,coco_gt)
            # if self.step % 1== 0:
                # self.tensor_board_log(imgs, labels,tt,kk)


if __name__ == '__main__':
    opt=get_params()
    trainer = Trainer(opt)
    trainer.train()
