import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from vicsgaze import VicsGaze
import torch
from torchvision import transforms
import numpy as np
import torchvision
from PIL import Image
import cv2
import os
import online_resnet
import torch.nn as nn

img_dir = "/root/autodl-tmp/MPIIFaceGaze-new/Image/p12/face"
visual_img = Image.open("/root/autodl-tmp/MPIIFaceGaze-new/Image/p02/face/1.jpg").convert('RGB')
# img_dir = "/root/autodl-tmp/Columbia/Image/train1"
# visual_img = Image.open("/root/autodl-tmp/Columbia/Image/train4/0045_2m_0P_0V_0H.jpg")

images=os.listdir(img_dir)

resnet = online_resnet.create_Res()
backbone = nn.Sequential(*list(resnet.children())[:-1])
net = VicsGaze(backbone)
model = nn.Sequential()
model.add_module('backbone', net.backbone)

statedict = torch.load("/root/autodl-tmp/vicsgaze/weights/Iter_40_vicsgaze_eth_ssl.pt")
# statedict = torch.load("/root/autodl-tmp/vicsgaze/result/eth/checkpoint/Iter_5_vicsgaze.pt")
# statedict = torch.load("/root/autodl-tmp/vicsgaze/result/mpii/checkpoint/p14.label/Iter_80_finetune.pt")
# statedict = torch.load("/root/autodl-tmp/vicsgaze/result/gaze360/checkpoint/Iter_80_finetune.pt")
# statedict = torch.load("/root/autodl-tmp/vicsgaze/result/columbia/checkpoint/train1.label/Iter_80_finetune.pt")


backbone_dict = {k: v for k, v in statedict.items() if 'backbone' in k}
model.load_state_dict(backbone_dict)
model.cuda()
# model = online_resnet.reparameterize_model(model)

input_H, input_W = 224, 224
heatmap = torch.zeros([input_H, input_W])

layer = model.backbone[-2][1].conv2
print(layer)

def farward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)
    

limit = 100
count = 0
for img in images:
    count += 1
    read_img = os.path.join(img_dir,img)
    image = Image.open(read_img)

    image = image.resize((input_H, input_W))
    image = np.float32(image) / 255
    input_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(image)

    input_tensor = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    input_tensor.requires_grad = True
    fmap_block = list()
    input_block = list()

    layer.register_forward_hook(farward_hook)

    output = model(input_tensor)
    feature_map = fmap_block[0].mean(dim=1,keepdim=False).squeeze()  # cnn

    # feature_map = fmap_block[0].mean(dim=2,keepdim=False).squeeze()  # vit
    # feature_map = feature_map.view(7, 7) # vit
    feature_map[(feature_map.shape[0]//2-1)][(feature_map.shape[1]//2-1)].backward(retain_graph=True)
    grad = torch.abs(input_tensor.grad)
    grad = grad.mean(dim=1, keepdim=False).squeeze()
    heatmap = heatmap + grad.cpu()
    if count >= limit:
        break
    
print("finish")
cam = heatmap
cam = cam.numpy()
cam = cam / cam.max()

cam = cv2.applyColorMap(np.uint8(cam*255), cv2.COLORMAP_JET)
image_array = np.array(visual_img)
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
result = cv2.addWeighted(np.uint8(image_array), 0.0, cam, 1.0, 0)
im = Image.fromarray(result)
im.save("cam.png")
