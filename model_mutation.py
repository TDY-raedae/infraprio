from yolo import YOLO

import torch
import os
import numpy as np
from PIL import Image
from utils.utils import (resize_image,preprocess_input)
import random
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
import tqdm
import pylab
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_weight_list={0:'backbone.conv1.weight',1:'backbone.layer1.ds_conv.weight',2:'backbone.layer1.residual_0.bn1.weight',3:'backbone.layer1.residual_0.conv2.weight',4:'backbone.layer2.ds_conv.weight',5:'backbone.layer2.residual_0.conv1.weight',6:'backbone.layer2.residual_0.conv2.weight',7:'backbone.layer2.residual_1.conv1.weight',8:'backbone.layer2.residual_1.conv2.weight',9:'backbone.layer3.ds_conv.weight',10:'backbone.layer3.residual_0.conv1.weight',11:'backbone.layer3.residual_0.conv2.weight',12:'backbone.layer3.residual_1.conv1.weight',13:'backbone.layer3.residual_1.conv2.weight',14:'backbone.layer3.residual_2.conv1.weight',15:'backbone.layer3.residual_2.conv2.weight',16:'backbone.layer3.residual_3.conv1.weight',17:'backbone.layer3.residual_3.conv2.weight',18:'backbone.layer3.residual_4.conv1.weight',19:'backbone.layer3.residual_4.conv2.weight',20:'backbone.layer3.residual_5.conv1.weight',21:'backbone.layer3.residual_5.conv2.weight',22:'backbone.layer3.residual_6.conv1.weight',23:'backbone.layer3.residual_6.conv2.weight',24:'backbone.layer3.residual_7.conv1.weight',25:'backbone.layer3.residual_7.conv2.weight',26:'backbone.layer4.ds_conv.weight',27:'backbone.layer4.residual_0.conv1.weight',28:'backbone.layer4.residual_0.conv2.weight',29:'backbone.layer4.residual_1.conv1.weight',30:'backbone.layer4.residual_1.conv2.weight',31:'backbone.layer4.residual_2.conv1.weight',32:'backbone.layer4.residual_2.conv2.weight',33:'backbone.layer4.residual_3.conv1.weight',34:'backbone.layer4.residual_3.conv2.weight',35:'backbone.layer4.residual_4.conv1.weight',36:'backbone.layer4.residual_4.conv2.weight',37:'backbone.layer4.residual_5.conv1.weight',38:'backbone.layer4.residual_5.conv2.weight',39:'backbone.layer4.residual_6.conv1.weight',40:'backbone.layer4.residual_6.conv2.weight',41:'backbone.layer4.residual_7.conv1.weight',42:'backbone.layer4.residual_7.conv2.weight',43:'backbone.layer5.ds_conv.weight',44:'backbone.layer5.residual_0.conv1.weight',45:'backbone.layer5.residual_0.conv2.weight',46:'backbone.layer5.residual_1.conv1.weight',47:'backbone.layer5.residual_1.conv2.weight',48:'backbone.layer5.residual_2.conv1.weight',49:'backbone.layer5.residual_2.conv2.weight',50:'backbone.layer5.residual_3.conv1.weight',51:'backbone.layer5.residual_3.conv2.weight',52:'last_layer0.0.conv.weight',53:'last_layer0.1.conv.weight',54:'last_layer0.2.conv.weight',55:'last_layer0.3.conv.weight',56:'last_layer0.4.conv.weight',57:'last_layer0.5.conv.weight',58:'last_layer1_conv.conv.weight',59:'last_layer1.0.conv.weight',60:'last_layer1.1.conv.weight',61:'last_layer1.2.conv.weight',62:'last_layer1.3.conv.weight',63:'last_layer1.4.conv.weight',64:'last_layer1.5.conv.weight',65:'last_layer2_conv.conv.weight',66:'last_layer2.0.conv.weight',67:'last_layer2.1.conv.weight',68:'last_layer2.2.conv.weight',69:'last_layer2.3.conv.weight',70:'last_layer2.4.conv.weight',71:'last_layer2.5.conv.weight'}

feature=[]
def cal_S(tensor1,tensor2):
    len=tensor2.flatten().shape[0]
    res=torch.abs(tensor1.flatten()-tensor2.flatten()).sum()/len
    return res
#定义钩子

def hook(model,input,output):
    feature.append(output.detach().cpu())
    return None
def sample_load(names,root):
    sample=[]
    for name in names:
        path=os.path.join(root,name)
        img=Image.open(path).convert('RGB')
        image_data = resize_image(img, (416, 416), False)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        image = torch.from_numpy(image_data)
        image = image.to(device)
        sample.append(image)
    return sample
def reset(weight):
    return torch.zeros_like(weight).to(device)

def gauss(weight):
    p=torch.randn(weight.shape).to(device)
    weight+=p
    return weight
length=[1,2,8,8,4]
def model_load(root,num):
    models=[]
    for i in range(num):
        if i==0:
            yolo = YOLO()
            model = yolo.net.to(device).eval()
            model.load_state_dict(torch.load('model_data/yolo_weights.pth', map_location=device))
            models.append(model)
        else:
            path=root+'m'+str(i-1)+'.pth'
            yolo = YOLO()
            model = yolo.net.to(device).eval()
            model.load_state_dict(torch.load(path, map_location=device))
            models.append(model)
    return models
def process(samples,flag=True,batch_size=32):
    result=[]
    cnt=0
    batch=None
    length=len(samples)
    for sample in samples:
        if flag==True:
            if cnt%batch_size!=0:
                batch=torch.cat([batch,sample.unsqueeze(0)],dim=0)
                if cnt==length-1:
                    result.append(batch.to(device))
            elif cnt%batch_size==0 or cnt>=length:
                if batch!=None and len(batch)>0:
                    result.append(batch.to(device))
                batch=sample.unsqueeze(0)
        else:
            if cnt%batch_size!=0:
                batch=torch.cat([batch,sample],dim=0)
                if cnt==length-1:
                    result.append(batch.to(device))
            elif cnt%batch_size==0 or cnt>=length:
                if batch!=None and len(batch)>0:
                    result.append(batch.to(device))
                batch=sample
        cnt+=1
    return result
def generate_mutated_yolo(ori_samples,mutation_0_samples,mutation_1_samples,mutation_2_samples,mutation_3_samples,mutation_4_samples):
    ori_samples = process(ori_samples,flag=False)
    mutation_0_samples = process(mutation_0_samples)
    mutation_1_samples = process(mutation_1_samples)
    mutation_2_samples = process(mutation_2_samples)
    mutation_3_samples = process(mutation_3_samples)
    mutation_4_samples = process(mutation_4_samples)
    print("finishing loading mutation data......")
    yolo = YOLO()
    model = yolo.net.to(device).eval()
    #设置hook
    yolo.net.backbone.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer1.ds_relu.register_forward_hook(hook)
    yolo.net.backbone.layer1.residual_0.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer1.residual_0.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer2.ds_relu.register_forward_hook(hook)
    yolo.net.backbone.layer2.residual_0.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer2.residual_0.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer2.residual_1.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer2.residual_1.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer3.ds_relu.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_0.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_0.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_1.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_1.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_2.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_2.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_3.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_3.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_4.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_4.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_5.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_5.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_6.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_6.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_7.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer3.residual_7.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer4.ds_relu.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_0.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_0.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_1.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_1.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_2.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_2.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_3.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_3.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_4.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_4.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_5.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_5.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_6.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_6.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_7.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer4.residual_7.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer5.ds_relu.register_forward_hook(hook)
    yolo.net.backbone.layer5.residual_0.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer5.residual_0.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer5.residual_1.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer5.residual_1.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer5.residual_2.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer5.residual_2.relu2.register_forward_hook(hook)
    yolo.net.backbone.layer5.residual_3.relu1.register_forward_hook(hook)
    yolo.net.backbone.layer5.residual_3.relu2.register_forward_hook(hook)
    yolo.net.last_layer0[0].relu.register_forward_hook(hook)
    yolo.net.last_layer0[1].relu.register_forward_hook(hook)
    yolo.net.last_layer0[2].relu.register_forward_hook(hook)
    yolo.net.last_layer0[3].relu.register_forward_hook(hook)
    yolo.net.last_layer0[4].relu.register_forward_hook(hook)
    yolo.net.last_layer0[5].relu.register_forward_hook(hook)
    yolo.net.last_layer1_conv.relu.register_forward_hook(hook)
    yolo.net.last_layer1[0].relu.register_forward_hook(hook)
    yolo.net.last_layer1[1].relu.register_forward_hook(hook)
    yolo.net.last_layer1[2].relu.register_forward_hook(hook)
    yolo.net.last_layer1[3].relu.register_forward_hook(hook)
    yolo.net.last_layer1[4].relu.register_forward_hook(hook)
    yolo.net.last_layer1[5].relu.register_forward_hook(hook)
    yolo.net.last_layer2_conv.relu.register_forward_hook(hook)
    yolo.net.last_layer2[0].relu.register_forward_hook(hook)
    yolo.net.last_layer2[1].relu.register_forward_hook(hook)
    yolo.net.last_layer2[2].relu.register_forward_hook(hook)
    yolo.net.last_layer2[3].relu.register_forward_hook(hook)
    yolo.net.last_layer2[4].relu.register_forward_hook(hook)
    yolo.net.last_layer2[5].relu.register_forward_hook(hook)

    sum=torch.zeros(72)
    #模型神经元激活值差异统计
    for i in tqdm.tqdm(range(len(ori_samples))):

        model(ori_samples[i])
        model(mutation_0_samples[i])
        model(mutation_1_samples[i])
        model(mutation_2_samples[i])
        model(mutation_3_samples[i])
        model(mutation_4_samples[i])

        mid_features=[]
        for j in range(6):
            temp=feature[j*72:(j+1)*72]
            mid_features.append(temp)

        feature.clear()
        for j in range(2,6):
            mid_features[1]+=mid_features[j]
        for j in range(len(mid_features[0])):
            sum[j]+=torch.abs(mid_features[0][j].mean(dim=0)-(mid_features[1][j]/5).mean(dim=0)).sum()/32
    #排序
    sort_result=sum.argsort()
    #根据下标确定变异目标神经元
    neurons=[]
    for i in range(5):
        neurons.append(yolo_weight_list[sort_result[len(sort_result)-1-i].item()])
    torch.save(model.state_dict(),'/home/Newdisk/yanyunjie/code_practics/patch/infrared_lab/mutation_result/mutation_model/ori_model.pth')
    for i in range(3):
        for neu in neurons:
            if i==0:
                model.state_dict()[neu]=gauss(model.state_dict()[neu])
            elif i==1:
                model.state_dict()[neu]*=-1
            elif i==2:
                model.state_dict()[neu]=reset(model.state_dict()[neu])
        torch.save(model.state_dict(),'/home/Newdisk/yanyunjie/code_practics/patch/infrared_lab/mutation_result/mutation_model/mutation_model_{}.pth'.format(i))
    print("finishing generating mutation models........................")

if __name__=='__main__':
    ori_samples = np.load('/home/Newdisk/yanyunjie/code_practics/patch/infrared_lab/mutation_result/mutation_images_ori.npy',allow_pickle=True)
    mutation_0_samples = np.load('/home/Newdisk/yanyunjie/code_practics/patch/infrared_lab/mutation_result/mutation_images_0.npy',allow_pickle=True)
    mutation_1_samples = np.load('/home/Newdisk/yanyunjie/code_practics/patch/infrared_lab/mutation_result/mutation_images_1.npy',allow_pickle=True)
    mutation_2_samples = np.load('/home/Newdisk/yanyunjie/code_practics/patch/infrared_lab/mutation_result/mutation_images_2.npy',allow_pickle=True)
    mutation_3_samples = np.load('/home/Newdisk/yanyunjie/code_practics/patch/infrared_lab/mutation_result/mutation_images_3.npy',allow_pickle=True)
    mutation_4_samples = np.load('/home/Newdisk/yanyunjie/code_practics/patch/infrared_lab/mutation_result/mutation_images_4.npy',allow_pickle=True)

    generate_mutated_yolo(ori_samples,mutation_0_samples,mutation_1_samples,mutation_2_samples,mutation_3_samples,mutation_4_samples)

