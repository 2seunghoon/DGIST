import torch
import torch.nn as nn
import torch.nn.functional as F
from mmd_loss import *

class Losses():
    def __init__(self,opt):
        self.opt = opt
        self.mmd_loss=MMD_loss()        
    def feature_distance(self, x1, x2, seg, classes, n_class=19):
        # import pdb;pdb.set_trace()

        fd_loss = 0.
        segmap = F.one_hot(seg+1, num_classes=n_class+1).permute(0,3,1,2)
        segmap = F.interpolate(segmap.float(), size=x1.size()[2:], mode='nearest') # [1,20,512,1024] size에서 -> [1,20,65,129]로 변경, 20은 class 수

        b_size = x1.shape[0]
        # h_size = x1.shape[2]
        # w_size = x1.shape[3]
        f_size = x1.shape[1]

        s_size = segmap.shape[1]

        x1_centroid = torch.zeros((b_size, s_size, f_size), dtype=x1.dtype, device=x1.device) # torch.Size([1, 20, 2048])
        x2_centroid = torch.zeros((b_size, s_size, f_size), dtype=x2.dtype, device=x2.device)
        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature1 = x1[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    x1_centroid[i][j] = codes_component_feature1
                    codes_component_feature2 = x2[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    x2_centroid[i][j] = codes_component_feature2
        x1_centroid = x1_centroid.mean(dim=0)[1:]
        x2_centroid = x2_centroid.mean(dim=0)[1:]
        
        for c in classes:
            # fd_loss += F.mse_loss(x1_centroid[c],x2_centroid[c]) # 승훈님이 주신 default code
            # fd_loss += F.l1_loss(x1_centroid[c],x2_centroid[c]) # testing l1_loss
            fd_loss+=self.mmd_loss(x2_centroid[c],x1_centroid[c])


        return fd_loss

def loss_weights(task, dsets):
    '''
        You must set these hyperparameters to apply our method to other datasets.
        These hyperparameters may not be the optimal value for your machine.
    '''

    alpha = dict()
    alpha['style'], alpha['dis'], alpha['gen'] = dict(), dict(), dict()
    if task == 'clf':
        alpha['recon'], alpha['consis'], alpha['content'] = 5, 1, 1

        # MNIST <-> MNIST-M
        if 'M' in dsets and 'MM' in dsets and 'U' not in dsets:
            alpha['style']['M2MM'], alpha['style']['MM2M'] = 5e4, 1e4
            alpha['dis']['M'], alpha['dis']['MM'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['MM'] = 0.5, 1.0

        # MNIST <-> USPS
        elif 'M' in dsets and 'U' in dsets and 'MM' not in dsets:
            alpha['style']['M2U'], alpha['style']['U2M'] = 5e3, 5e3
            alpha['dis']['M'], alpha['dis']['U'] = 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['U'] = 0.5, 0.5

        # MNIST <-> MNIST-M <-> USPS
        elif 'M' in dsets and 'U' in dsets and 'MM' in dsets:
            alpha['style']['M2MM'], alpha['style']['MM2M'], alpha['style']['M2U'], alpha['style']['U2M'] = 5e4, 1e4, 1e4, 1e4
            alpha['dis']['M'], alpha['dis']['MM'], alpha['dis']['U'] = 0.5, 0.5, 0.5
            alpha['gen']['M'], alpha['gen']['MM'], alpha['gen']['U'] = 0.5, 1.0, 0.5

    elif task == 'seg':
        # GTA5 <-> Cityscapes
        alpha['recon'], alpha['consis'], alpha['content'] = 10, 1, 1
        alpha['style']['G2C'], alpha['style']['C2G'] = 5e3, 5e3
        alpha['dis']['G'], alpha['dis']['C'] = 0.5, 0.5
        alpha['gen']['G'], alpha['gen']['C'] = 0.5, 0.5

    return alpha


class Loss_Functions:
    def __init__(self, args):
        self.args = args
        self.alpha = loss_weights(args.task, args.datasets)

    def recon(self, imgs, recon_imgs):
        recon_loss = 0
        for dset in imgs.keys():
            recon_loss += F.l1_loss(imgs[dset], recon_imgs[dset])
        return self.alpha['recon'] * recon_loss
        
    def dis(self, real, fake):
        dis_loss = 0
        if self.args.task == 'clf':  # DCGAN loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.binary_cross_entropy(real[dset], torch.ones_like(real[dset]))
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.binary_cross_entropy(fake[cv], torch.zeros_like(fake[cv]))
        elif self.args.task == 'seg':  # Hinge loss
            for dset in real.keys():
                dis_loss += self.alpha['dis'][dset] * F.relu(1. - real[dset]).mean()
            for cv in fake.keys():
                source, target = cv.split('2')
                dis_loss += self.alpha['dis'][target] * F.relu(1. + fake[cv]).mean()
        return dis_loss

    def gen(self, fake):
        gen_loss = 0
        for cv in fake.keys():
            source, target = cv.split('2')
            if self.args.task == 'clf':
                gen_loss += self.alpha['gen'][target] * F.binary_cross_entropy(fake[cv], torch.ones_like(fake[cv]))
            elif self.args.task == 'seg':
                gen_loss += -self.alpha['gen'][target] * fake[cv].mean()
        return gen_loss

    def content_perceptual(self, perceptual, perceptual_converted):
        content_perceptual_loss = 0
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            content_perceptual_loss += F.mse_loss(perceptual[source][-1], perceptual_converted[cv][-1])
        return self.alpha['content'] * content_perceptual_loss

    def style_perceptual(self, style_gram, style_gram_converted):
        style_percptual_loss = 0
        for cv in style_gram_converted.keys():
            source, target = cv.split('2')
            for gr in range(len(style_gram[target])):
                style_percptual_loss += self.alpha['style'][cv] * F.mse_loss(style_gram[target][gr], style_gram_converted[cv][gr])
        return style_percptual_loss

    def consistency(self, contents, styles, contents_converted, styles_converted, converts):
        consistency_loss = 0
        for cv in converts:
            source, target = cv.split('2')
            consistency_loss += F.l1_loss(contents[cv], contents_converted[cv])
            consistency_loss += F.l1_loss(styles[target], styles_converted[cv])
        return self.alpha['consis'] * consistency_loss

    def task(self, pred, gt):
        task_loss = 0
        for key in pred.keys():
            if '2' in key:
                source, target = key.split('2')
            else:
                source = key
            task_loss += F.cross_entropy(pred[key], gt[source], ignore_index=-1)
        return task_loss

