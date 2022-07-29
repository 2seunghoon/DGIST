
# -*- coding: utf-8 -*-
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import copy
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms
import cv2
from ...OWDA.dataloader.Cityscapes import Cityscapes
from albumentations.augmentations.transforms import ColorJitter
from things_stuff_transfer import *
import skimage
ImageFile.LOAD_TRUNCATED_IMAGES = True

class GTA5_3(Cityscapes):
    def __init__(self,
                 list_path='./data_list/GTA5',
                 split='train',
                 crop_size=(512, 256),
                 train=True,
                 numpy_transform=False
                 ):

        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True

        image_list_filepath = os.path.join(self.list_path, self.split + "_imgs.txt")
        label_list_filepath = os.path.join(self.list_path, self.split + "_labels.txt")

        image_list_filepath_imgnet=os.path.join('./data_list/imagenet-mini', self.split + "_imgs.txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        self.labels = [id.strip() for id in open(label_list_filepath)]
        self.images_imgnet=[id.strip() for id in open(image_list_filepath_imgnet)]
        
        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        print("{} num images in GTA5 {} set have been loaded.".format(len(self.images), self.split))

    def __getitem__(self, item):
        
        ## imagenet 무작위 2개 추출
        # num_lines = sum(1 for line in open('/home/cvintern2/Desktop/intern/OWDA/data_list/imagenet-mini/train_imgs.txt'))
        # item_imgnet=np.random.randint(0,num_lines,size=2)
        item_imgnet=np.random.randint(0,34745,size=2)
        ## select random
        option=np.random.randint(0,34745,size=1)
        ##
        image_path = self.images[item]
        image_path_imgnet_1=self.images_imgnet[item_imgnet[0]]
        image_path_imgnet_2=self.images_imgnet[item_imgnet[1]]

        image = Image.open(image_path).convert("RGB")
        image_cv=cv2.imread(image_path)
        imagenet_cv=self.images_imgnet[item]
        imagenet_cv=cv2.imread(imagenet_cv)
        imagenet_pil=Image.open(self.images_imgnet[item]).convert("RGB")
        image_imgnet_1=Image.open(image_path_imgnet_1).convert("RGB")
        image_imgnet_2=Image.open(image_path_imgnet_2).convert("RGB")


        gt_image_path = self.labels[item]
        gt_image = Image.open(gt_image_path)

        # things stuff change
        #transfer=things_stuff_change(image,image_imgnet_1,image_imgnet_2,np.array(gt_image)) # stuff : imgnet1, things : imgnet2
        #transfer=cv2.cvtColor(transfer,cv2.COLOR_BGR2RGB)
        #transfer=Image.fromarray(transfer)

        # gaussian noise
        gaussian_image = skimage.io.imread(image_path)
        gaussian_image = skimage.util.random_noise(gaussian_image, mode='gaussian')
        gaussian_image=Image.fromarray(gaussian_image.astype('uint8'))
        ##

        ## class 별 aug
        class_aug=class_things_stuff_change(image,np.array(gt_image)) # stuff : imgnet1, things : imgnet2
        class_aug=cv2.cvtColor(class_aug.astype('uint8'),cv2.COLOR_BGR2RGB)
        class_aug=Image.fromarray(class_aug)

        ## things stuff transfer aug
        things_stuff=things_stuff_change(image,image_imgnet_1,image_imgnet_2,np.array(gt_image)) # stuff : imgnet1, things : imgnet2
        things_stuff=cv2.cvtColor(things_stuff.astype('uint8'),cv2.COLOR_BGR2RGB)
        things_stuff=Image.fromarray(things_stuff)

        # 기존 Color Transfer
        ## 여기서 gta + imagenet 합쳐줄거
        c_transfer = color_transfer(imagenet_cv, image_cv) # color transfer input [1046,1914,3] =[H,W,C]
        # # 합친후 RGB로 변환 PIL에 맞도록
        c_transfer=cv2.cvtColor(c_transfer,cv2.COLOR_BGR2RGB)
        c_transfer=Image.fromarray(c_transfer)


        # resize 사이즈 키우기
        # image_imgnet=cv2.resize(image_imgnet,(256,512))
        # image_imgnet_1=cv2.cvtColor(np.array(image_imgnet_1),cv2.COLOR_BGR2RGB)
        # image_imgnet_1=Image.fromarray(image_imgnet_1)
        # image_imgnet_2=cv2.cvtColor(np.array(image_imgnet_2),cv2.COLOR_BGR2RGB)
        # image_imgnet_2=Image.fromarray(image_imgnet_2)

        ## for ColorJitter
        cj=ColorJitter(always_apply=True,p=1)
        color_jitter=cj(image=image_cv)
        color_jitter['image']=cv2.cvtColor(color_jitter['image'],cv2.COLOR_BGR2RGB)
        color_jitter=Image.fromarray(color_jitter['image'])
        ##
        ###


        # cv2.imwrite('origin1.jpg',image_cv)
        # cv2.imwrite('image1.jpg',image_imgnet)
        # transfer.save('synthesis.jpg')
        # gt_image.save('GT.jpg')
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            image, gt_image = self._train_sync_transform(image, gt_image)
            
            # For final
            color_jitter= self._train_sync_transform(color_jitter,None)
            gaussian_image=self._train_sync_transform(gaussian_image,None)
            things_stuff=self._train_sync_transform(things_stuff,None)
            class_aug=self._train_sync_transform(class_aug,None)
            c_transfer=self._train_sync_transform(c_transfer,None)
            # image=self._train_sync_transform(image, None)
            # ## For Jitter
            imagenet_pil=self._train_sync_transform(imagenet_pil,None)

        ## 여기서 synthesis이미지도 transform해줄거
            # transfer=self._train_sync_transform(transfer, None)

            # image_imgnet_1=self._train_sync_transform(image_imgnet_1, None)
            # image_imgnet_2=self._train_sync_transform(image_imgnet_2, None)

            
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

            # ## For Jitter
            # color_jitty, gt_image = self._val_sync_transform(color_jitty, gt_image)
            # image = self._val_sync_transform(image, None)
            # ##
            color_jitter= self._val_sync_transform(color_jitter,None)
            gaussian_image=self._val_sync_transform(gaussian_image,None)
            things_stuff=self._val_sync_transform(things_stuff,None)
            class_aug=self._val_sync_transform(class_aug,None)
            c_transfer=self._val_sync_transform(c_transfer,None)

            # image_imgnet_1 = self._val_sync_transform(image_imgnet_1, None)
            # image_imgnet_2 = self._val_sync_transform(image_imgnet_2, None)
            imagenet_pil=self._val_sync_transform(imagenet_pil,None)


       
        ## return 3개할거, image(gta),gt_image,synthesis(gta+imagenet),
        # return image, gt_image,transfer

        ## for Jitter
        # return color_jitty, gt_image,transfer,image
        ##
        # 원래 코드
        # return image, gt_image,transfer,image_imgnet_1,image_imgnet_2
        # class별 aug
        # return image, gt_image,transfer

        # things_stuff
        return image, gt_image,imagenet_pil,gaussian_image,things_stuff,class_aug,color_jitter,c_transfer




