from yolo import YOLO
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
ToTensor=transforms.Compose([transforms.ToTensor()])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image
#载入模型
yolo=YOLO()
model=yolo.net.to(device).eval()
model.load_state_dict(torch.load('logs/best_epoch_weights.pth', map_location=device))
img=Image.open('img/target.jpg').convert('RGB')
img=resize_image(img,(416,416),False)
image=ToTensor(img).unsqueeze(0).cuda()
x=model(image)
