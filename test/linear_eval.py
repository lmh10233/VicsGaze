import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
from models.Gaze_regressor import Gaze_regressor
import random
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse
import torchvision
from loss import NegativeCosineSimilarity
from warmup_scheduler import GradualWarmupScheduler
import online_resnet
# from byol import BYOL
# from simsiam import SimSiam
from vicsgaze import VicsGaze
import time


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    
def main(train, test):
    # seed_torch()
    
    fine_tune = True
    total_training_time = 0  # 初始化总训练时间

    # resnet = torchvision.models.resnet18()
    resnet = online_resnet.create_Res()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    # net = BYOL(backbone)
    # net = SimSiam(backbone)
    net = VicsGaze(backbone)
    model = nn.Sequential()
    model.add_module('backbone', net.backbone)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_reader = importlib.import_module("reader." + train.reader)
    test_reader = importlib.import_module("reader." + test.reader)
    torch.cuda.set_device(test.device)

    # params = test.params
    data = test.data
    load = test.load
    
    regressor = Gaze_regressor()
    regressor.train(); regressor.cuda()
    
    train_params = test.params_linear
    train_data = test.train_data
    train_save = test.save_linear
    
    if data.isFolder: 
        data, _ = ctools.readfolder(data) 
    
    test_dataset = test_reader.loader(data, 32, num_workers=4, shuffle=False)
    # modelpath = os.path.join(train.save.metapath,
                             # train.save.folder, f"checkpoint/")
    modelpath = os.path.join("weights")
    
    logpath = os.path.join(train.save.metapath,
                                train.save.folder, f"{test.savename}")

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    statedict = torch.load(
            os.path.join(modelpath, 
                f"Iter_{train.params.epoch}_{train.save.model_name}_eth_s.pt"), 
            map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
        )
    backbone_dict = {k: v for k, v in statedict.items() if 'backbone' in k}
    # backbone_dict = {k: v for k, v in statedict.items() if 'backbone' in k and 'backbone_momentum' not in k and 'momentum_backbone' not in k}
    model.to(device);  model.load_state_dict(backbone_dict)
    
    if train_data.isFolder:
        train_data, _ = ctools.readfolder(train_data)

    train_dataset = train_reader.loader(
                    train_data,
                    train_params.batch_size, 
                    shuffle=True, 
                    num_workers=8,
                    fine_tune=fine_tune
                )

    if fine_tune:
        print("-- [mode]: finetune_eval")
        parameters = list(model.parameters()) + list(regressor.parameters())
        # optimizer = optim.SGD(
        #     parameters,
        #     lr=train_params.lr, 
        #     momentum=0.9)
        optimizer = optim.Adam(
            parameters, 
            lr=train_params.lr, 
            betas=(0.9, 0.95))
    else:
        print("-- [mode]: linear_eval")
        for param in model.parameters():
            param.requires_grad = False  
            
        # optimizer = optim.SGD(
        #     regressor.parameters(),
        #     lr=train_params.lr, 
        #     momentum=0.9)
        optimizer = optim.Adam(
            regressor.parameters(), 
            lr=train_params.lr, 
            betas=(0.9, 0.95))
        
    scheduler = optim.lr_scheduler.StepLR( 
                    optimizer, 
                    step_size=train_params.decay_step, 
                    gamma=train_params.decay
                )
    
    if train_params.warmup:
        scheduler = GradualWarmupScheduler( 
                        optimizer, 
                        multiplier=1, 
                        total_epoch=train_params.warmup, 
                        after_scheduler=scheduler
                    )

    train_savepath = os.path.join(train_save.metapath, train_save.folder, f"checkpoint")

    if not os.path.exists(train_savepath):
        os.makedirs(train_savepath)
    
    length = len(train_dataset); total = length * train_params.epoch
    timer = ctools.TimeCounter(total)
    
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()
    
    # 训练回归器
    with open(os.path.join(train_savepath, "train_linear_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(train) + '\n')
        for epoch in range(1, train_params.epoch + 1):
            start_time = time.time()  # 训练开始计时
            for i, (data, anno) in enumerate(train_dataset):
                # -------------- forward -------------
                for key in data:
                    if key != 'name': data[key] = data[key].cuda()
                anno = anno.cuda()
                # with torch.no_grad():
                feature = model(data["face"])
                feature = feature.flatten(start_dim=1)
                loss = regressor.loss(feature, anno)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                rest = timer.step()/3600
                if i % 20 == 0:
                    log = f"[{epoch}/{train_params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"loss:{loss} " +\
                          f"lr:{ctools.GetLR(optimizer)} " + \
                          f"rest time:{rest:.2f}h"

                    print(log); outfile.write(log + "\n")
                    sys.stdout.flush(); outfile.flush()
            
            scheduler.step()
            end_time = time.time()
            batch_time = end_time - start_time  # 计算每批次的时间
            total_training_time += batch_time
            
            if epoch % train_save.step == 0:
                torch.save(
                        regressor.state_dict(), 
                        os.path.join(
                            train_savepath, 
                            f"Iter_{epoch}_{train_save.model_name}.pt"
                            )
                        )
                if fine_tune:
                    torch.save(model.state_dict(), 
                            os.path.join(
                                train_savepath, 
                                f"Iter_{epoch}_finetune.pt"
                                ))
        second_per_batch = total_training_time / train_params.epoch+1
        print(f"Second per Batch: {second_per_batch} seconds per batch")
      
    begin = load.begin_step; end = load.end_step; step = load.steps
    
    regressorpath = os.path.join(test.save_linear.metapath,
                             test.save_linear.folder, f"checkpoint/")
    
    for saveiter in range(begin, end+step, step):
        # 测试性能
        print(f"Test {saveiter}")
        
        regressor_statedict = torch.load(
            os.path.join(regressorpath, 
                f"Iter_{saveiter}_{test.save_linear.model_name}.pt"), 
            map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
        )
        regressor = Gaze_regressor()
        regressor.cuda(); regressor.load_state_dict(regressor_statedict); regressor.eval(); 

        model = nn.Sequential()
        model.add_module('backbone', net.backbone)
        if fine_tune:
            print("-- [model adding]: finetune load model parameters")
            finetune_statedict = torch.load(
                os.path.join(regressorpath, 
                    f"Iter_{saveiter}_finetune.pt"), 
                map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
            )

            model.load_state_dict(finetune_statedict); model.eval(); model = online_resnet.reparameterize_model(model)

        logname = f"{saveiter}.log"
        outfile = open(os.path.join(logpath, logname), 'w')
        outfile.write("name results gts\n")

        length = len(test_dataset); accs = 0; count = 0
        with torch.no_grad():
            for j, (data, label) in enumerate(test_dataset):

                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                names =  data["name"]
                # data = data["face"]
                gts = label.cuda()

                feature = model(data["face"])
                feature = feature.flatten(start_dim=1)
                gazes = regressor(feature).cuda()

                for k, gaze in enumerate(gazes):

                    gaze = gaze.cpu().detach().numpy()
                    gt = gts.cpu().numpy()[k]

                    count += 1                
                    accs += gtools.angular(
                                gtools.gazeto3d(gaze),
                                gtools.gazeto3d(gt)
                            )

                    name = [names[k]]
                    gaze = [str(u) for u in gaze] 
                    gt = [str(u) for u in gt] 
                    log = name + [",".join(gaze)] + [",".join(gt)]
                    outfile.write(" ".join(log) + "\n")

            loger = f"[{saveiter}] Total Num: {count}, avg: {accs/count}"
            outfile.write(loger)
            print(loger)
        outfile.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about test')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    test_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    main(train_conf.train, test_conf.test)
    
    
