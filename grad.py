from yolo import YOLO
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import Adam
from utils.utils import (resize_image,preprocess_input)
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from nets.yolo_training import YOLOLoss
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#将Image转换为tensor
ToTensor=transforms.Compose([transforms.ToTensor(),])
unloader=transforms.ToPILImage()
features=[]
def hook(module,input,ouput):
    features.append(input)
    return None
def grad_f(model,image_data):

    # 重构形状

    # 做转换为tensor的准备工作


    x=Variable(image_data,requires_grad=True)
    outs = model(x)

    sum=0
    grad_total=0
    #求取平均值
    for idx in range(3):
        temp=torch.sum(outs[idx]**2)
        temp.backward(retain_graph=True)
        grad=x.grad.squeeze(0)
        grad_s=grad[0]+grad[1]+grad[2]
        average=torch.sum(grad_s)/(416*416*3)
        sum+=average
        grad_total+=grad_s/3
    aver=sum/3
    grad_total/=3


    mask=torch.where(grad_total-300>=aver,1,0)
    mask=mask.unsqueeze(0)
    mask=mask.repeat(3,1,1)
    return mask,image_data
if __name__=='__main__':

    #读入图片
    path='/home/Newdisk/yanyunjie/code_practics/patch/yolov5/2dsample/0.png'
    mask,image=grad_f(path)
    img = image.squeeze(0)
    img_m = 1-img.cpu() * mask
    p = unloader(img_m)
    plt.imshow(p)
    plt.axis('off')
    plt.show()





