import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
def things_stuff_change(gta_image,imgnet1,imgnet2,gt):
    ## gta image=1052,1914,3 # PIL
    # imgnet = 333,500,3 # PIL
    # gt= 1052,1914 # np

    ## stuff -> imgnet1 color
    ## things -> imgnet2 color
    
    # resize
    if gt.shape!=np.array(gta_image).shape[0:2]:
        gta_image = gta_image.resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)
    # if mask: mask = mask.resize(self.crop_size, Image.NEAREST)
    ##
   
   
    # mask_stuff_class=[0,1,7,8,11,12,13,17,19,20,21,22,23]
    mask_stuff_class=[-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,29,30] # ignore 고려

    mask_things_class=[24,25,26,27,28,31,32,33]
    
    gt=np.array(gt)

    # 통합
    mask_stuff=np.zeros(np.shape(gt))
    mask_things=np.zeros(np.shape(gt))
    for i in mask_stuff_class:
        mask_stuff+=np.array(gt==i)
    for i in mask_things_class:
        mask_things+=np.array(gt==i)
    
    mask_stuff=torch.Tensor(mask_stuff).unsqueeze(0)
    mask_stuff=torch.cat([mask_stuff,mask_stuff,mask_stuff],dim=0)
    mask_things=torch.Tensor(mask_things).unsqueeze(0)
    mask_things=torch.cat([mask_things,mask_things,mask_things],dim=0)

    masked_stuff_gta=mask_stuff*(np.array(gta_image).transpose(2,0,1))
    masked_things_gta=mask_things*(np.array(gta_image).transpose(2,0,1))

    output_things=color_transfer(cv2.cvtColor(np.array(imgnet2),cv2.COLOR_RGB2BGR),cv2.cvtColor(np.array(masked_things_gta).transpose(1,2,0).astype('uint8'),cv2.COLOR_RGB2BGR))

    masked_things_gta=mask_things*(np.array(output_things).transpose(2,0,1))
    # plt.imshow(np.array(masked_things_gta).transpose(1,2,0).astype('uint8'))

    output_stuff=color_transfer(cv2.cvtColor(np.array(imgnet1),cv2.COLOR_RGB2BGR),cv2.cvtColor(np.array(masked_stuff_gta).transpose(1,2,0).astype('uint8'),cv2.COLOR_RGB2BGR))

    masked_stuff_gta=mask_stuff*(np.array(output_stuff).transpose(2,0,1))
    # plt.imshow(np.array(masked_stuff_gta).transpose(1,2,0).astype('uint8'))

    transfered=np.array(masked_stuff_gta).transpose(1,2,0).astype('uint8')+np.array(masked_things_gta).transpose(1,2,0).astype('uint8')
    # transfered=transfered.transpose(2,0,1)
    # (1046, 1914, 3) numpy
    return transfered
def class_things_stuff_change(gta_image,gt):
    ## gta image=1052,1914,3 # PIL
    # imgnet = 333,500,3 # PIL
    # gt= 1052,1914 # np

    ## stuff -> imgnet1 color
    ## things -> imgnet2 color
    
    # resize
    if gt.shape!=np.array(gta_image).shape[0:2]:
        gta_image = gta_image.resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)
    # if mask: mask = mask.resize(self.crop_size, Image.NEAREST)
    ##
   
   
    # mask_stuff_class=[0,1,7,8,11,12,13,17,19,20,21,22,23]
    mask_stuff_class=[-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,29,30] # ignore 고려

    mask_things_class=[24,25,26,27,28,31,32,33]
    
    gt=np.array(gt)
    tot=[]
    for i in np.unique(gt):
        if i in mask_stuff_class+mask_things_class:
            tot.append(i)
    unique_class=len(tot)
    

    # class별
    # masked_stuff=[ gt[:,:] ==i for i in mask_stuff_class ]
    # masked_things=[ gt[:,:] ==i for i in mask_things_class ]

    item_imgnet=np.random.randint(0,34745,size=unique_class)
    image_list_filepath_imgnet=os.path.join('./data_list/imagenet-mini', 'train' + "_imgs.txt")
    images_imgnet=[id.strip() for id in open(image_list_filepath_imgnet)]
    imgnet_path=[]
    for i in range(len(item_imgnet)):
        imgnet_path.append(images_imgnet[item_imgnet[i]])
    mask_by_class=[]
    for i in range(len(item_imgnet)):
        mask=np.array(gt==tot[i])
        mask=torch.Tensor(mask).unsqueeze(0)
        mask=torch.cat([mask,mask,mask],dim=0)
        mask_gta=mask*(np.array(gta_image).transpose(2,0,1))
        imgnet=Image.open(imgnet_path[i]).convert('RGB')
        output=color_transfer(cv2.cvtColor(np.array(imgnet),cv2.COLOR_RGB2BGR),cv2.cvtColor(np.array(mask_gta).transpose(1,2,0).astype('uint8'),cv2.COLOR_RGB2BGR))
        mask_by_class.append(mask*(np.array(output).transpose(2,0,1)))

    transfered=np.zeros([gt.shape[0],gt.shape[1],3])
    for i in range(len(item_imgnet)):
        transfered+=np.array(mask_by_class[i]).transpose(1,2,0).astype('uint8')
    # transfered=transfered.transpose(2,0,1)
    # (1046, 1914, 3) numpy

    return transfered

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
