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
from .Cityscapes import Cityscapes
from albumentations.augmentations.transforms import ColorJitter

ImageFile.LOAD_TRUNCATED_IMAGES = True

class GTA5(Cityscapes):
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

        image_path = self.images[item]
        image_path_imgnet=self.images_imgnet[item]
        image = Image.open(image_path).convert("RGB")
        image_cv=cv2.imread(image_path)
        image_imgnet=cv2.imread(image_path_imgnet)


        gt_image_path = self.labels[item]
        gt_image = Image.open(gt_image_path)

        ## 여기서 gta + imagenet 합쳐줄거
        import pdb;pdb.set_trace()
        transfer = color_transfer(image_imgnet, image_cv) # color transfer input [1046,1914,3] =[H,W,C]
        # 합친후 RGB로 변환 PIL에 맞도록
        transfer=cv2.cvtColor(transfer,cv2.COLOR_BGR2RGB)
        transfer=Image.fromarray(transfer)
        # resize 사이즈 키우기
        # image_imgnet=cv2.resize(image_imgnet,(256,512))
        image_imgnet=cv2.cvtColor(image_imgnet,cv2.COLOR_BGR2RGB)
        image_imgnet=Image.fromarray(image_imgnet)


        ## for ColorJitter
        cj=ColorJitter(p=1)
        color_jitty=cj(image=image_cv)
        color_jitty['image']=cv2.cvtColor(color_jitty['image'],cv2.COLOR_BGR2RGB)
        color_jitty=Image.fromarray(color_jitty['image'])
        ##
        ###


        # cv2.imwrite('origin1.jpg',image_cv)
        # cv2.imwrite('image1.jpg',image_imgnet)
        # transfer.save('synthesis.jpg')
        # gt_image.save('GT.jpg')
        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            # image, gt_image = self._train_sync_transform(image, gt_image)
            
            ## For Jitter
            color_jitty, gt_image = self._train_sync_transform(color_jitty, gt_image)
            image=self._train_sync_transform(image, None)
            ## For Jitter


        ## 여기서 synthesis이미지도 transform해줄거
            transfer=self._train_sync_transform(transfer, None)
            image_imgnet=self._train_sync_transform(image_imgnet, None)
            
        else:
            # image, gt_image = self._val_sync_transform(image, gt_image)

            ## For Jitter
            color_jitty, gt_image = self._val_sync_transform(color_jitty, gt_image)
            image = self._val_sync_transform(image, None)
            ##

            transfer = self._val_sync_transform(transfer, None)
            image_imgnet = self._val_sync_transform(image_imgnet, None)


       
        ## return 3개할거, image(gta),gt_image,synthesis(gta+imagenet),
        # return image, gt_image,transfer

        ## for Jitter
        return color_jitty, gt_image,transfer,image
        ##
        
        # 원래 코드
        # return image, gt_image,transfer,image_imgnet



def color_transfer(source, target, clip=True, preserve_paper=True):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    source: NumPy array
        OpenCV image in BGR color space (the source image)
    target: NumPy array
        OpenCV image in BGR color space (the target image)
    clip: Should components of L*a*b* image be scaled by np.clip before 
        converting back to BGR color space?
        If False then components will be min-max scaled appropriately.
        Clipping will keep target image brightness truer to the input.
        Scaling will adjust image brightness to avoid washed out portions
        in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
        layed out in original paper? The method does not always produce
        aesthetically pleasing results.
        If False then L*a*b* components will scaled using the reciprocal of
        the scaling factor proposed in the paper.  This method seems to produce
        more consistently aesthetically pleasing results 

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l = (lStdSrc / lStdTar) * l
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    
    # return the color transferred image
    return transfer

def image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space

    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array

    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array

    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled

def _scale_array(arr, clip=True):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.

    Parameters:
    -------
    arr: array to be trimmed to [0, 255] range
    clip: should array be scaled by np.clip? if False then input
        array will be min-max scaled to range
        [max([arr.min(), 0]), min([arr.max(), 255])]

    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled
