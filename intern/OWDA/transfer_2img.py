from __future__ import print_function
from random import seed
from logging import Formatter, StreamHandler, getLogger, FileHandler
from tkinter import Image
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

from things_stuff_transfer import *

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

    def set_optimizers(self):
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

    def train_task(self, imgs, labels,transfer):  # Train Task Networks (T)

        self.set_zero_grad()
        *_, GTA_feature = self.nets['T'](imgs[self.source], lbl=labels[self.source])
        # *_, City_feature = self.nets['T'](imgs[self.target])
        loss_seg_src_origin = self.nets['T'].loss_seg

        *_, GTA_feature = self.nets['T'](transfer[self.source], lbl=labels[self.source])
        loss_seg_src_synthesis = self.nets['T'].loss_seg

        # *_, ImageNet_feature = self.nets['ImageNet'](imgs[self.source], lbl=labels[self.source])
        # feature_distance = self.loss_fns.feature_distance(GTA_feature, ImageNet_feature, labels[self.source], self.ImageNet_classes) 
        loss_task = loss_seg_src_origin+loss_seg_src_synthesis
        loss_task.backward()
        self.optims['T'].step()
        self.losses['T'] = loss_seg_src_origin.data.item()
        self.losses['synthesis'] = loss_seg_src_synthesis.data.item()

        # self.losses['FD'] = feature_distance.data.item()

    def tensor_board_log(self, imgs, labels,transfer,imgnet_1,imgnet_2):
        nrow = 5
        ##
        ##
        # Input Images & Recon Images
        
        for dset in self.opt.datasets:
            x = vutils.make_grid(imgs[dset].detach(), normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('1_Input_Images/%s' % dset, x, self.step)
        ## transfer
        x = vutils.make_grid(transfer['G'].detach(), normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('1_Input_Images/%s' % 'Transfer', x, self.step)
        x = vutils.make_grid(imgnet_1['G'], normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('1_Input_Images/%s' % 'Stuff', x, self.step)
        x = vutils.make_grid(imgnet_2['G'], normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('1_Input_Images/%s' % 'Things', x, self.step)
        ##     ##
        # Segmentation GT, Pred
        self.set_eval()
        preds = dict()
        trans_preds=dict()

        for dset in self.opt.datasets:
            x = decode_labels(labels[dset].detach(), num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('4_GT/%s' % dset, x, self.step)
            preds[dset] = self.nets['T'](imgs[dset])[0]
            if dset=='G':
                trans_preds[dset]=self.nets['T'](transfer[dset])[0]
        for key in preds.keys():
            pred = preds[key].data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            x = decode_labels(pred, num_images=self.opt.batch)
            x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
            self.writer.add_image('5_Prediction/%s' % key, x, self.step)
        
        ## for transfer
        key='G'
        trans_pred = trans_preds[key].data.cpu().numpy()
        trans_pred = np.argmax(trans_pred, axis=1)
        x = decode_labels(trans_pred, num_images=self.opt.batch)
        x = vutils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
        self.writer.add_image('5_Prediction/%s' % 'Transfer', x, self.step)        
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
            '[%d/%d] Origin: %.2f|Synthesis : %.2f| %.2f %s'
            % (self.step, self.opt.iter,
                self.losses['T'],self.losses['synthesis'], self.best_miou, self.opt.ex))

    def train(self):
        self.set_default()
        self.set_networks()
        self.set_optimizers()
        self.set_train()
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
            

            for dset in self.opt.datasets:
                if dset=='G':
                    # imgs[dset], labels[dset],imgnet[dset]= batch_data[dset]
                    imgs[dset], labels[dset],transfer[dset],imgnet_1[dset],imgnet_2[dset]= batch_data[dset]
                    imgs[dset], labels[dset],transfer[dset] = imgs[dset].cuda(), labels[dset].cuda(),transfer[dset].cuda()
                    labels[dset] = labels[dset].long()
                elif dset=='C':
                    imgs[dset], labels[dset]= batch_data[dset]
                    imgs[dset], labels[dset] = imgs[dset].cuda(), labels[dset].cuda()
                    labels[dset] = labels[dset].long()                   
            # training

            self.train_task(imgs, labels,transfer)
            if self.step % 10 == 0:
                self.tensor_board_log(imgs, labels,transfer,imgnet_1,imgnet_2)
            if self.step % self.opt.eval_freq == 0:
                for target in self.targets:
                    self.eval(target)
            self.print_loss()


if __name__ == '__main__':
    opt=get_params()
    trainer = Trainer(opt)
    trainer.train()
