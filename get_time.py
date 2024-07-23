import torch
import torchvision
# from simsiam import SimSiam
from vicreg import VicReg
import time
import torch.nn as nn
import online_resnet
from models.Gaze_regressor import Gaze_regressor

iterations = 500   # 重复计算的轮次

resnet = torchvision.models.resnet18()
# resnet = online_resnet.create_Res()
backbone = nn.Sequential(*list(resnet.children())[:-1])
net = VicReg(backbone)
model = nn.Sequential()
model.add_module('backbone', net.backbone)
model.eval(); model = online_resnet.reparameterize_model(model)

regressor = Gaze_regressor()
regressor.eval()

device = torch.device("cuda")
model.to(device); regressor.to(device)

random_input = torch.randn(1, 3, 224, 224).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(100):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        x = model(random_input)
        x = x.flatten(start_dim=1)
        x = regressor(x)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)
# with torch.no_grad():
#     for iter in range(iterations):
#         start = time.time()
#         _ = model(random_input)
#         end = time.time()
#         times[iter] = (end - start)*1000
#         # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
