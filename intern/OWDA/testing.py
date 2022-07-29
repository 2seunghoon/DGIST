print("\n... IMPORTS STARTING ...\n")

print("\n\tVERSION INFORMATION")

# Machine Learning and Data Science Imports
import pandas as pd; pd.options.mode.chained_assignment = None;
import numpy as np; print(f"\t\t– NUMPY VERSION: {np.__version__}");
import sklearn; print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}");
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scipy.spatial import cKDTree

# # RAPIDS
# import cudf, cupy, cuml
# from cuml.neighbors import NearestNeighbors
# from cuml.manifold import TSNE, UMAP
# from cuml import PCA

# Built In Imports
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import warnings
import requests
import hashlib
import imageio
import IPython
import sklearn
import urllib
import zipfile
import pickle
import random
import shutil
import string
import json
import math
import time
import gzip
import ast
import sys
import io
import os
import gc
import re

# Visualization Imports
import plotly

from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm; tqdm.pandas();
import plotly.express as px
import tifffile as tif
from PIL import Image, ImageEnhance; Image.MAX_IMAGE_PIXELS = 5_000_000_000;
import matplotlib; print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}");
from matplotlib import animation, rc; rc('animation', html='jshtml')
import plotly
import PIL
import cv2
import tensorflow as tf

import plotly.io as pio
print(pio.renderers)
def get_imagenet1000mini_df(dir_path,mode, add_shape_info=False):

    # Create dataframe as none exists natively
    # _df = pd.DataFrame({"img_path":tf.io.gfile.glob(os.path.join(dir_path,"**","*.JPEG"))})
    # _df = pd.DataFrame({"img_path":tf.io.gfile.glob(os.path.join(dir_path,"**","*.JPEG"))})
    img_path=list()
    if mode=='train':
        filepath='/home/cvintern2/Desktop/intern/OWDA/data_list/imagenet-mini/train_imgs.txt'
        filelist=tf.io.gfile.glob(os.path.join(dir_path,"**","*.JPEG"))
        with open(filepath,'w+') as lf:
            for i in range(len(filelist)):
                img_path.append(filelist[i].split('/OWDA/')[1])
            lf.write('\n'.join(img_path))
    else:
        filepath='/home/cvintern2/Desktop/intern/OWDA/data_list/imagenet-mini/val_imgs.txt'
        filelist=tf.io.gfile.glob(os.path.join(dir_path,"**","*.JPEG"))
        with open(filepath,'w+') as lf:
            for i in range(len(filelist)):
                img_path.append(filelist[i].split('/OWDA/')[1])
            lf.write('\n'.join(img_path))
    # Cleanup
    gc.collect(); gc.collect(); gc.collect()
    
get_imagenet1000mini_df(dir_path="/home/cvintern2/Desktop/intern/OWDA/data/imagenet-mini/train",mode='train')
get_imagenet1000mini_df(dir_path="/home/cvintern2/Desktop/intern/OWDA/data/imagenet-mini/val",mode='val')
