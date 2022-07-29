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

class filtering():
    def __init__(self):
        self.coco = COCO('../data/coco/annotations/instances_train2017.json')
        self.class_txt = {
            24: "../class_txt/person.txt",
            26: "../class_txt/car.txt",
            27: "../class_txt/truck.txt",
            28: "../class_txt/bus.txt",
            31: "../class_txt/train.txt",
            33: "../class_txt/bicycle.txt",
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
        self.images_coco_dic = {}
        for self.class_number in [19,24,26,27,28,31,32,33]:
            class_label = self.mask_labels[self.class_number]
            catIds = self.coco.getCatIds(catNms=[class_label])
            imgIds = self.coco.getImgIds(catIds=catIds)
            self.images_coco_dic[self.class_number] = self.coco.loadImgs(imgIds)
    def get_img_ann(self):
        self.class_number = 24
        with open(self.class_txt[self.class_number]) as file1:
            Lines = file1.readlines()
        for line in Lines:
            print(line)
            # self.class_number = 24# random.choice([19,24,26,27,28,31,32,33])
            # print("self.class_number", self.class_number)
            self.class_label = self.mask_labels[self.class_number]
            if self.class_number != 19 and self.class_number != 32:
                self.rand_number_in_class = int(line)# random.randint(0, len(Lines))
                images = self.images_coco_dic[self.class_number][self.rand_number_in_class]
            else:
                images = random.choice(self.images_coco_dic[self.class_number])

            # 이미지 가져오기
            self.coco_img_pil = Image.open(requests.get(images['coco_url'], stream=True).raw).convert('RGB')


            self.coco_img_np = np.empty((self.coco_img_pil.size[1], self.coco_img_pil.size[0]))
            self.coco_empty_pil = Image.fromarray(self.coco_img_np)
            # Annotation 가져오기
            annIds = self.coco.getAnnIds(imgIds=images['id'], catIds=self.catIds_mapping[self.class_number], iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            # mask 하나만 골라오기
            self.coco_mask_np = self.coco.annToMask(anns[0])
            ###########################################################
            self.coco_mask_pil = Image.fromarray(self.coco_mask_np)
            # self.coco_mask_np_255 = self.coco_mask_np*255
            # self.coco_mask_np_class = self.coco_mask_np * self.class_number
            # self.coco_mask_pil_class = Image.fromarray(self.coco_mask_np_class)
            # self.coco_mask_pil = Image.fromarray(self.coco_mask_np_255)

            self.coco_empty_pil.paste(self.coco_img_pil, (0, 0), self.coco_mask_pil)


            self.coco_empty_pil.save("check.png")
            good_or_bad = input('good or bad')
            if good_or_bad == "y":
                with open("person_filter.txt", "a") as f:
                    f.write(line)
        return

f = filtering()
f.get_img_ann()