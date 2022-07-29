# -*- coding: utf-8 -*-
import random
from re import X
import scipy.io
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import copy
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms
import time
from .Cityscapes import Cityscapes
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import traceback
from pycocotools.coco import COCO
import os
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor, ToPILImage

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import masks_to_boxes
# instantiate COCO specifying the annotations json path
# coco = COCO('./cocodataset/annotations/instances_train2017.json')
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

class F_model(Cityscapes):
    def __init__(self,
                 list_path='./data_list/GTA5',
                 split='train',
                 crop_size=(1024, 512),
                 train=True,
                 numpy_transform=False,
                 opt=None
                 ):
        self.coco = COCO('./data/coco/annotations/instances_train2017.json')
        self.list_path = list_path
        self.split = split
        self.crop_size = crop_size
        self.train = train
        self.numpy_transform = numpy_transform
        self.resize = True
        self.opt = opt
        self.iterrr = 0
        image_list_filepath = os.path.join(self.list_path, self.split + "_imgs.txt")
        label_list_filepath = os.path.join(self.list_path, self.split + "_labels.txt")

        if not os.path.exists(image_list_filepath):
            raise Warning("split must be train")

        self.images = [id.strip() for id in open(image_list_filepath)]
        self.labels = [id.strip() for id in open(label_list_filepath)]

        ignore_label = -1
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.mask_labels = {
            19: 'traffic light',
            24: 'person',
            26: 'car',
            27: 'truck',
            28: 'bus',
            31: 'train',
            32: 'motorcycle',
            33: 'bicycle',
        }
        print("{} num images in GTA5 {} set have been loaded.".format(len(self.images), self.split))
        # Step 1: Initialize model with the best available weights
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.eval()

        # Step 2: Initialize the inference transforms
        self.preprocess = self.weights.transforms()

        self.class_percent = {
            19: np.load("./class_percent/traffic light.npy"),
            24: np.load("./class_percent/person.npy"),
            26: np.load("./class_percent/car.npy"),
            27: np.load("./class_percent/truck.npy"),
            28: np.load("./class_percent/bus.npy"),
            31: np.load("./class_percent/train.npy"),
            32: np.load("./class_percent/motorcycle.npy"),
            33: np.load("./class_percent/bicycle.npy"),
        }
        self.class_txt = {
            24: "./class_txt/person.txt",
            26: "./class_txt/car.txt",
            27: "./class_txt/truck.txt",
            28: "./class_txt/bus.txt",
            31: "./class_txt/train.txt",
            33: "./class_txt/bicycle.txt",
        }
        self.catIds_mapping = {
            19: 10,
            24: 1,
            26: 3,
            27: 8,
            28: 6,
            31: 7,
            32: 4,
            33: 2
        }
        self.images_coco_dic = {}
        for self.class_number in [19,24,26,27,28,31,32,33]:
            class_label = self.mask_labels[self.class_number]
            catIds = self.coco.getCatIds(catNms=[class_label])
            imgIds = self.coco.getImgIds(catIds=catIds)
            self.images_coco_dic[self.class_number] = self.coco.loadImgs(imgIds)

    def __getitem__(self, item):
        image_path = self.images[item]
        image_ori = Image.open(image_path).convert("RGB")

        # image.save(f'image_{item}.jpg')

        gt_image_path = self.labels[item]
        gt_image_ori = Image.open(gt_image_path)
        # gt_image.save(f'gt_image_{item}.jpg')

        color_transfered_image, gt_image, coco_img, coco_gt  = self.copy_paste(image_ori, gt_image_ori, item)

        # image.save("image.png")
        # gt_image.save("gt_image.png")

        if (self.split == "train" or self.split == "trainval" or self.split =="all") and self.train:
            image, gt_image = self._train_sync_transform(color_transfered_image, gt_image)
            coco_img, coco_gt =self._train_sync_transform(coco_img, coco_gt)
        else:
            image, gt_image = self._val_sync_transform(image_ori, gt_image_ori)
            coco_img, coco_gt = self._val_sync_transform(coco_img, coco_gt)

        return image, gt_image, coco_img, coco_gt

    def copy_paste(self, image_ori, gt_image_ori, index_num):
        self.image_ori = image_ori
        self.gt_image_ori = gt_image_ori
        # gt와 img의 사이즈가 다른 사진 62장이 있어서 그것 핸들링
        if image_ori.size != gt_image_ori.size:
            image_ori = image_ori.resize((gt_image_ori.size[0], gt_image_ori.size[1]), Image.NEAREST)
            with open("log.txt", "a") as f:
                msg = f"image_ori.size: {image_ori.size} gt_image_ori.size: {gt_image_ori.size} index_num: {index_num}\n"
                f.write(msg)

        # cropped 된 coco 사진 붙여넣기
        self.image_ori_np_3d = np.array(image_ori)
        self.gt_image_ori_np_1d = np.array(gt_image_ori)

        self.get_img_ann()

        # COCO 이미지 crop 하기
        self.crop_object()
        # COCO 이미지 classification 하기
        # self.while_classify()


        # self.coco_bool_masks_1d = self.cropped_coco_mask_np == 1
        # self.coco_bool_masks_1d_expand = np.expand_dims(self.coco_bool_masks_1d, 2)
        # self.coco_bool_masks_3d = np.concatenate((self.coco_bool_masks_1d_expand, self.coco_bool_masks_1d_expand, self.coco_bool_masks_1d_expand), 2)

        # COCO 이미지 위치 넣기
        self.output_img, self.output_gt = self.place_object()

        return self.output_img, self.output_gt, self.coco_img_pil, self.coco_mask_pil_class

    def get_img_ann(self):
        # class random 하게 고르기
        while True:
            try:
                self.class_number = random.choice([19,24,26,27,28,31,32,33])
                # self.class_number=26
                # print("self.class_number", self.class_number)
                self.class_label = self.mask_labels[self.class_number]
                if self.class_number != 19 and self.class_number != 32:
                    with open(self.class_txt[self.class_number]) as file1:
                        Lines = file1.readlines()
                    self.rand_number_in_class = random.randint(0, len(Lines))
                    images = self.images_coco_dic[self.class_number][self.rand_number_in_class]
                else:
                    images = random.choice(self.images_coco_dic[self.class_number])

                # 이미지 가져오기
                self.coco_img_pil = Image.open(requests.get(images['coco_url'], stream=True).raw).convert('RGB')
                break
            except:
                pass
        
        
        
        self.coco_img_np = np.array(self.coco_img_pil)

        # Annotation 가져오기
        annIds = self.coco.getAnnIds(imgIds=images['id'], catIds=self.catIds_mapping[self.class_number], iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        # mask 하나만 골라오기
        self.coco_mask_np = self.coco.annToMask(anns[0])
        for i in range(1, len(anns)):
            self.coco_mask_np += self.coco.annToMask(anns[i])
        ###########################################################
        self.coco_mask_np_255 = self.coco_mask_np*255
        self.coco_mask_np_class = self.coco_mask_np * self.class_number
        self.coco_mask_pil_class = Image.fromarray(self.coco_mask_np_class)
        self.coco_mask_pil = Image.fromarray(self.coco_mask_np_255)
        # self.save_img_np_to_pil(self.coco_img_np, "coco_img_np.png")

        return
    # def classify(self, img_np):
    #     tf_toTensor = ToTensor() 
    #     img = tf_toTensor(img_np)
    #     # Step 3: Apply inference preprocessing transforms
    #     batch = self.preprocess(img).unsqueeze(0)

    #     # Step 4: Use the model and print the predicted category
    #     prediction = self.model(batch).squeeze(0).softmax(0)
    #     class_id = prediction.argmax().item()
    #     score = prediction[class_id].item()
    #     category_name = self.weights.meta["categories"][class_id]
    #     # import pdb; pdb.set_trace()
    #     # print(f"{category_name}: {100 * score:.1f}%")
    #     return 100 * score
    # def while_classify(self):
    #     while True:
    #         classify_score = self.classify(self.cropped_coco_img_np)
    #         if classify_score>=1:
    #             break
    #         else:
    #             self.get_img_ann()
    #             self.crop_object()
            # self.coco_bool_masks_1d = self.cropped_coco_mask_np == 1
            # self.coco_bool_masks_1d_expand = np.expand_dims(self.coco_bool_masks_1d, 2)
            # self.coco_bool_masks_3d = np.concatenate((self.coco_bool_masks_1d_expand, self.coco_bool_masks_1d_expand, self.coco_bool_masks_1d_expand), 2)
            # self.only_coco_img_np = np.zeros(self.coco_img_np.shape, dtype=np.uint8)
            # self.only_coco_img_np[self.coco_bool_masks_3d] = self.coco_img_np[self.coco_bool_masks_3d]
            # self.save_img_np_to_pil(self.only_coco_img_np, "only_coco_img_np.png")


    def save_img_np_to_pil(self, img, path):
        pil_img = Image.fromarray(img)
        pil_img.save(path)
        return

    def crop_object(self):
        # coco_img_np,  coco_mask_np input으로 들어오고
        # output은 작은 사이즈의 small_coco_img_np, small_coco_mask_np
        tf_toTensor = ToTensor()
        mask_tensor = tf_toTensor(self.coco_mask_np)
        boxes = masks_to_boxes(mask_tensor)
        boxes_np = boxes.numpy().astype(np.uint16)

        
        self.cropped_coco_img_np = self.coco_img_np[boxes_np[0][1]:boxes_np[0][3], boxes_np[0][0]:boxes_np[0][2]]
        self.cropped_coco_mask_np_255 = self.coco_mask_np_255[boxes_np[0][1]:boxes_np[0][3], boxes_np[0][0]:boxes_np[0][2]]
        self.cropped_coco_mask_np_class = self.coco_mask_np_class[boxes_np[0][1]:boxes_np[0][3], boxes_np[0][0]:boxes_np[0][2]]
        # import pdb; pdb.set_trace()
        # np.unique(self.cropped_coco_mask_np_class)

        self.cropped_coco_img_pil = Image.fromarray(self.cropped_coco_img_np)
        self.cropped_coco_mask_pil = Image.fromarray(self.cropped_coco_mask_np_255)
        self.cropped_coco_mask_pil_class = Image.fromarray(self.cropped_coco_mask_np_class)



        # self.cropped_coco_img_pil = self.coco_img_pil.crop((boxes_np[0][0], boxes_np[0][1], boxes_np[0][2], boxes_np[0][3]))
        # self.cropped_coco_mask_pil = self.coco_mask_pil.crop((boxes_np[0][0], boxes_np[0][1], boxes_np[0][2], boxes_np[0][3]))
        # self.cropped_coco_mask_pil_class = self.coco_mask_pil_class.crop((boxes_np[0][0], boxes_np[0][1], boxes_np[0][2], boxes_np[0][3]))

        # random resize
        # choicelist = random.choice([1])
        if self.class_number == 19:
            scale = 1
        elif self.cropped_coco_img_pil.size[0] < 50:
            scale = 5
        elif self.cropped_coco_img_pil.size[0] < 100:
            scale = 4
        elif self.cropped_coco_img_pil.size[0] < 200:
            scale = 3
        # scale = self.image_ori.size[0] / self.cropped_coco_img_pil.size[0] / 5
        else:
            scale = 1
        # print(self.cropped_coco_img_pil.size[0], "scale: ",scale)
        self.cropped_coco_img_pil = self.cropped_coco_img_pil.resize((int(self.cropped_coco_img_pil.size[0]*scale), int(self.cropped_coco_img_pil.size[1]*scale)), Image.NEAREST)
        self.cropped_coco_mask_pil = self.cropped_coco_mask_pil.resize((int(self.cropped_coco_mask_pil.size[0]*scale), int(self.cropped_coco_mask_pil.size[1]*scale)), Image.NEAREST)
        self.cropped_coco_mask_pil_class = self.cropped_coco_mask_pil_class.resize((int(self.cropped_coco_mask_pil_class.size[0]*scale),int(self.cropped_coco_mask_pil_class.size[1]*scale)), Image.NEAREST)



        # self.cropped_coco_img_pil.paste(self.cropped_coco_img_pil, (0, 0), self.cropped_coco_mask_pil)
        # self.cropped_coco_img_pil.save("self.cropped_coco_img_pil.png")
        ######################################################################################################################
        self.cropped_coco_img_np = np.array(self.cropped_coco_img_pil)
        self.cropped_coco_mask_np = np.array(self.cropped_coco_mask_pil)
        self.cropped_coco_mask_np_class = np.array(self.cropped_coco_mask_pil_class)

        # self.coco_bool_masks_1d = self.cropped_coco_mask_np == 1
        # self.coco_bool_masks_1d_expand = np.expand_dims(self.coco_bool_masks_1d, 2)
        # self.coco_bool_masks_3d = np.concatenate((self.coco_bool_masks_1d_expand, self.coco_bool_masks_1d_expand, self.coco_bool_masks_1d_expand), 2)
        # self.coco_bool_masks_1d_inv = ~self.coco_bool_masks_1d
        # self.coco_bool_masks_3d_inv = ~self.coco_bool_masks_3d

        return

    def place_object(self):
        # print(self.mask_labels[self.class_number])
        percent_np = self.class_percent[self.class_number]
        pixelnum = np.random.choice(np.arange(len(percent_np)), p=percent_np)
        # print("pixelnum", pixelnum)
        x_start_position = int(pixelnum%self.image_ori.size[0])
        y_start_position = int((pixelnum-x_start_position)/self.image_ori.size[0])
        x_start_position = int(x_start_position  - self.cropped_coco_img_pil.size[0]/2)
        y_start_position = int(y_start_position - self.cropped_coco_img_pil.size[1]/2)
        # print(x_start_position)
        # print(y_start_position)
        # source_img_cv = cv2.cvtColor(self.cropped_coco_img_np, cv2.COLOR_RGB2BGR) #cv2.imread("00003_gt.png")
        # target_img_cv = cv2.cvtColor(self.image_ori_np_3d, cv2.COLOR_RGB2BGR) #cv2.imread("image.png")
        # bool, output_img = color_transfer(source_img_cv, target_img_cv)
        # if not bool:
        #     output_img = target_img_cv
        # color_converted_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        pil_img = self.image_ori
        back_ground_im = pil_img.copy()
        back_ground_gt = self.gt_image_ori.copy()
        self.cropped_coco_mask_pil_blur = self.cropped_coco_mask_pil.filter(ImageFilter.GaussianBlur(10))
        # back_ground_im.paste(self.cropped_coco_img_pil, (x_start_position, y_start_position), self.cropped_coco_mask_pil_blur)
        # back_ground_gt.paste(self.cropped_coco_mask_pil_class, (x_start_position, y_start_position), self.cropped_coco_mask_pil)
        # back_ground_gt.save("BBBBB.png")
        # np.unique(np.array(self.cropped_coco_mask_pil_class))
        # radius = 500
        # image_draw = ImageDraw.Draw(back_ground_im)
        # image_draw.ellipse((x_start_position - radius, y_start_position - radius, x_start_position + radius, y_start_position + radius), fill=(0, 0, 255))
        return back_ground_im, back_ground_gt

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
    if aStdSrc == 0:
        return False, None
    if bStdSrc == 0:
        return False, None
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
    return True, transfer

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