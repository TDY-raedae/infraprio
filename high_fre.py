import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import os
from utils.utils import (resize_image,preprocess_input)

ToTensor=transforms.Compose([transforms.ToTensor()])
unloader=transforms.ToPILImage()
path='/home/NewDisk/yanyunjie/re/yolo3-pytorch-master/VOCdevkit/VOC2007/JPEGImages/'

D=50
radius_ratio=0.5
def generate_Fourier_mask(img,radius_ratio):
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频
    template=np.ones(fshift.shape,np.uint8)
    row,col=int(fshift.shape[0]/2),int(fshift.shape[1]/2)
    radius=int(radius_ratio*img.shape[0]/2)
    cv2.circle(template,(row,col),radius,0,-1)
    hight_parts_fshift =  template*fshift
    ishift = np.fft.ifftshift(hight_parts_fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)
    high_parts_img = iimg
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                    np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    thres, img_bin=cv2.threshold(img_new_high,25,255,cv2.THRESH_BINARY)
    mask=ToTensor(img_bin/255.0).type(torch.float32)
    return mask
if __name__=='__main__' :
    mask=generate_Fourier_mask('img/sample3.jpg',0.5)

