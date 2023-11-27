from high_fre import generate_Fourier_mask
from grad import grad_f
from yolo import YOLO
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def guassian(mask,img):
    mean=0
    sigma=1
    guassian=torch.from_numpy(np.random.normal(mean,sigma,img.shape).astype(np.float32)).to(device)
    img_g=torch.where(mask>0,img+guassian,img)
    return img_g
def whiten(mask,img):
    tensor=torch.ones(img.shape).to(device)
    img_w=torch.where(mask>0,tensor,img)
    return img_w
def blacken(mask,img):
    tensor=torch.zeros(img.shape).to(device)
    img_b=torch.where(mask>0,tensor,img)
    return img_b
def grey_change(mask,img):
    temp=torch.where(mask>0,img,0)
    temp1=torch.where(temp>0.5,temp-0.5,0)
    temp2=torch.where(temp<0.5,temp+0.5,0)
    temp3=temp1+temp2
    result=torch.where(mask>0,temp3,img)
    return result
def pixel(mask,img):
    temp=torch.ones_like(img).to(device)
    temp[:,1:,1:]*=img[:,:415,:415]
    print(temp.shape,img.shape)
    result=torch.where(mask>0,temp,img)
    return result


def reshape_image(image,size):
    iw,ih=image.size
    w,h=size
    scale=min(w/iw,h/ih)
    nw=int(iw*scale)
    nh=int(ih*scale)

    image=image.resize((nw,nh),Image.BICUBIC)
    new_image=Image.new('RGB',size,(128,128,128))
    new_image.paste(image,((w-nw)//2,(h-nh)//2))
    return new_image
#筛选进行变异的特征点

def filter(img,mask,rate,bin):
    one=torch.ones_like(mask[0]).cuda()
    zero=torch.zeros_like(mask[0]).cuda()
    mask1=torch.where(bin>rate,one,zero)*mask[0]

    mask2=torch.where(bin>rate,one,zero)*mask[0]
    mask3 = torch.where(bin > rate, one, zero) * mask[0]
    mask4 = torch.where(bin > rate, one, zero)* mask[0]
    mask5 = torch.where(bin > rate, one, zero) * mask[0]
    return mask1.unsqueeze(0).repeat(3,1,1).cuda(),mask2.unsqueeze(0).repeat(3,1,1).cuda(),mask3.unsqueeze(0).repeat(3,1,1).cuda(),mask4.unsqueeze(0).repeat(3,1,1).cuda(),mask5.cuda()#输出的掩膜形状都是(3,416,416)
def mutation(img,mask,index):
    if index==0:
        img=guassian(mask,img)
    elif index==1:
        img=whiten(mask,img)
    elif index==2:
        img=blacken(mask,img)
    elif index==3:
        img = grey_change(mask, img)
    elif index==4:
        img=pixel(mask,img)
    return img
unloader=transforms.ToPILImage()

def show_mask(mask):
    img=unloader(mask)
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.show()
if __name__=='__main__':
    root="./img0/"
    #加载检测模型
    flag=True
    yolo = YOLO()
    model = yolo.net.to(device).eval()
    model.load_state_dict(torch.load('logs/best_epoch_weights.pth', map_location=device))
    print("model loading finished.....")
    ori_list=[]
    for i in range(5):
        list=[]
        for name in os.listdir(root):
            path=os.path.join(root,name)
            img = cv2.imread(path, 0)
            img = cv2.resize(img, dsize=(416, 416))
            mask1=generate_Fourier_mask(img,0.5).repeat(3,1,1).to(device)
            image=torch.from_numpy(np.array(img, dtype='float32')/255.0).unsqueeze(0).repeat(3,1,1).unsqueeze(0).to(device)
            mask2,image=grad_f(model,image)
            if flag==True:
                ori_list.append(image.cpu())
            image=image.squeeze(0)
            mask=(mask1+mask2).clamp(0,1)
            image_mutation=mutation(image,mask,i)
            list.append(image_mutation.cpu())
        mutation_result=np.asarray(list)
        np.save("./mutation_result/mutation_images_{}.npy".format(i),mutation_result,allow_pickle=True)
        if flag==True:
            np.save("./mutation_result/mutation_images_ori.npy",np.asarray(ori_list),allow_pickle=True)
            flag=False
    print("datasets finished.....")
