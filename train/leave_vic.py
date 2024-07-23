import sys, os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import random
import copy
import yaml
import cv2
import copy
import torch
import torchvision
from torch import nn
import importlib
import ctools
import argparse
import time
import numpy as np
from loss import NegativeCosineSimilarity
from vics_loss.loss import VicsLoss
# from byol import BYOL
# from simsiam import SimSiam
from vicsgaze import VicsGaze
from warmup_scheduler import GradualWarmupScheduler
from scheduler import cosine_schedule
from easydict import EasyDict as edict
import online_resnet

def seed_torch(seed=3407):  # 114514, 3407
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config):
    seed_torch()
    # resnet = torchvision.models.resnet18()
    resnet = online_resnet.create_Res()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    # model = BYOL(backbone)
    # model = SimSiam(backbone)
    model = VicsGaze(backbone)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    total_training_time = 0  # 初始化总训练时间
    #  ===================>> Setup <<=================================

    # We disable resizing and gaussian blur for cifar10.

    dataloader = importlib.import_module("reader." + config.reader)
    
    data = config.data
    save = config.save
    params = config.params
    
    data, folder = ctools.readfolder(
                        data, 
                        # [config.person], 
                        reverse=True
                    )
    
    savename = folder[config.person] 
        
    dataset = dataloader.loader(
                    data,
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=16,
                )
    
    # criterion = NegativeCosineSimilarity()
    citeration = VicsLoss()
    
    # parameter = list(model.backbone.parameters()) + list(model.projection_head.parameters()) + list(model.prediction_head.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, betas=(0.9, 0.95), weight_decay=params.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, weight_decay=params.weight_decay)
    # optimizer = LARS(parameter, lr=params.lr, weight_decay=params.weight_decay, momentum=0.9)
  
    scheduler = torch.optim.lr_scheduler.StepLR( 
                optimizer, 
                step_size=params.decay_step, 
                gamma=params.decay
            )
    if params.warmup:
        scheduler = GradualWarmupScheduler( 
                        optimizer, 
                        multiplier=1, 
                        total_epoch=params.warmup, 
                        after_scheduler=scheduler
                    )
    
    savepath = os.path.join(save.metapath, save.folder, f'checkpoint/{savename}')

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    length = len(dataset); total = length * params.epoch
    timer = ctools.TimeCounter(total)
    
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()
    
    print("Starting Training")
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')
        for epoch in range(1, params.epoch + 1):
            total_loss = 0
            # momentum_val = cosine_schedule(epoch, params.epoch, 0.99, 1)
            for i, (batch, anno) in enumerate(dataset):
                data = batch['face']
                x0 = data[0]
                x1 = data[1]
                x0 = x0.to(device)
                x1 = x1.to(device)
                z0 = model(x0)
                z1 = model(x1)
                # loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
                loss = citeration(z0, z1)
                total_loss += loss.detach()
                step_loss = loss.detach()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                rest = timer.step()/3600
                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"lr:{ctools.GetLR(optimizer)} " +\
                          f"loss:{step_loss} " +\
                          f"rest time:{rest:.2f}h"

                    print(log); outfile.write(log + "\n")
                    sys.stdout.flush(); outfile.flush()
                
            scheduler.step()
            
            if (epoch) % save.step == 0:
                torch.save(
                        model.state_dict(), 
                        os.path.join(
                            savepath, 
                            f"Iter_{epoch}_{save.model_name}.pt"
                            )
                        )
            
            avg_loss = total_loss / len(dataset)
            log = f"epoch:{epoch}" + f"[{avg_loss}] "
            outfile.write(log + "\n")
            sys.stdout.flush(); outfile.flush()
            print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')

    parser.add_argument('-p', '--person', type=int,
                        help='The tested person.')

    args = parser.parse_args()
    
    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))
    
    config = config.train
    config.person = args.person
    main(config)