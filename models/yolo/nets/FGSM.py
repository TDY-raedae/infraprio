import torch
import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import os
from PIL import Image
import sys
sys.path.append('/home/NewDisk/jinhaibo/lxhtest/re/yolo3-pytorch-master')
from yolo import YOLO
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VOCDataset(torch.utils.data.Dataset):

    CLASSES_NAME = (
        "__background__ ",                 # 记得加上背景类
        'person', 'bike', 'car', 'motor', 'bud', 'train','truck','light','hydrant','sign','dog','deer','skateboard','stroller','scooter','parking meter'
    )
    # 初始化类
    def __init__(self, root_dir, resize_size=[800, 1024], split='trainval', use_difficult=False):

        self.root = root_dir
        self.use_difficult = use_difficult
        self.imgset = split

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        # 读取trainval.txt中内容
        with open(self._imgsetpath % self.imgset) as f:     # % 是python字符串中的一个转义字符可以百度下，不难
            self.img_ids = f.readlines()
        self.img_ids = [x.strip() for x in self.img_ids]    # ['000009', '000052']

        self.name2id = dict(zip(VOCDataset.CLASSES_NAME, range(len(VOCDataset.CLASSES_NAME))))
        self.resize_size = resize_size
        self.mean = [0.485, 0.456, 0.406]      # voc数据集中所有图像矩阵的均值和方差，为后续图像归一化做准备
        self.std = [0.229, 0.224, 0.225]
        print("INFO=====>voc dataset init finished  ! !")

    def __len__(self):
        return len(self.img_ids)

    def _read_img_rgb(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):

        img_id = self.img_ids[index]
        img = self._read_img_rgb(self._imgpath % img_id)

        anno  = ET.parse(self._annopath % img_id).getroot()  # 读取xml文档的根节点
        boxes = []
        classes = []

        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            _box = obj.find("bndbox")
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE = 1                                  # 由于像素是网格存储，坐标2实质表示第一个像素格，所以-1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name = obj.find("name").text.lower().strip()
            classes.append(self.name2id[name])             # 将类别映射回去

        boxes = np.array(boxes, dtype=np.float32)

        #将img,box和classes转成tensor
        img = transforms.ToTensor()(img)    # transforms 自动将 图像进行了归一化，
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes
# if __name__ == '__main__':
#%%
dataset = VOCDataset('/home/NewDisk/yanyunjie/re/yolo3-pytorch-master/VOCdevkit/VOC2007') # 实例化一个对象
img,box,cls = dataset[0]          # 返回第一张图像及box和对应的类别
print(img.shape)
print(box)
print(cls)

# 这里简单做一下可视化
# 由于opencv读入是矩阵，而img现在是tensor，因此，首先将tensor转成numpy.array
img_ = (img.numpy()*255).astype(np.uint8).transpose(1,2,0)# 注意由于图像像素分布0-255，所以转成uint8
print(img_.shape)
# cv2.imshow('test',img_)
# cv2.waitKey(0)

#%%
yolo = YOLO()
model = yolo.net.to(device).eval()
model.load_state_dict(torch.load('/home/NewDisk/yanyunjie/re/yolo3-pytorch-master/logs/best_epoch_weights.pth', map_location=device))
#%%
img_in = Image.open('/home/NewDisk/yanyunjie/re/yolo3-pytorch-master/img/aaa.png')
image_shape = np.array(np.shape(img_in)[0:2])
image = cvtColor(img_in)

image_data = resize_image(img_in, (416, 416), False)
image_data.save("/home/NewDisk/yanyunjie/re/yolo3-pytorch-master/img/img_ori.jpg")
image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
images = torch.from_numpy(image_data)
images = images.cuda()
criterion = nn.MSELoss()
x_adv = Variable(images.data, requires_grad=True)
h_adv = model(x_adv)
confidence = h_adv[0][0][4] ###改 输出是三个feature 可以都试试一起优化，我这里只选了一个看看效果

#%%
false_label = np.zeros(confidence.shape)
false_label = torch.tensor(false_label).to(device)
top_label_adv = np.array(results_adv[0][:, 6], dtype = 'int')
top_label_adv = torch.tensor(top_label_adv).to(device)
cost = criterion(h_adv[0][0][4].double(), false_label.double()) ##损失函数

model.zero_grad()
cost.backward() ##梯度
#%%
x_adv.grad.sign_() ##符号
x_adv = x_adv - 0.03 * x_adv.grad
x_adv = torch.clamp(x_adv, 0, 1)
x_ori = Variable(images.data, requires_grad=True)
tt = x_adv.cpu().detach().numpy()

#%%
plt.imshow(tt[0].transpose(1, 2, 0))
plt.show()
#%%
cv2.imwrite('/home/NewDisk/yanyunjie/re/yolo3-pytorch-master/img/aaa_adv.png', tt[0].transpose(1, 2, 0) * 255)
#%%
crop = False
count = False
image = Image.open('/home/NewDisk/yanyunjie/re/yolo3-pytorch-master/img/aaa_adv.png')
r_image = yolo.detect_image(image, crop=crop, count=count)
r_image.save("/home/NewDisk/yanyunjie/re/yolo3-pytorch-master/img/img_adv.jpg")
