#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project      : MaskedRoads
# @File         : baseline.py
# @Desc         :
# @Author       : Chengxin
# @CreateTime   : 2020/10/29 下午3:55

# Import lib here
from unet import UNet
from utile import deeplearning as dl
import os
import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000000000
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 指定路径，构建数据集
data_dir = "/home/chenxin/Datasets/MaskedRoads/data/"
train_imgs_dir = os.path.join(data_dir, "train/images/")
val_imgs_dir = os.path.join(data_dir, "val/images/")
train_labels_dir = os.path.join(data_dir, "train/labels/")
val_labels_dir = os.path.join(data_dir, "val/labels/")
train_data = dl.RSCDataset(train_imgs_dir, train_labels_dir)
valid_data = dl.RSCDataset(val_imgs_dir, val_labels_dir)
checkpoint_dir = os.path.join("/home/chenxin/Projects/Competition/MaskedRoads/ckpt/", 'unet/') # 模型保存路径
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

# 模型，参数，训练
model = UNet(3, 2).to(device)
# 参数设置
param = {}
param['epochs'] = 41       # 训练轮数
param['batch_size'] = 4   # 批大小
param['lr'] = 2e-2         # 学习率
param['gamma'] = 0.9       # 学习率衰减系数
param['step_size'] = 5     # 学习率衰减间隔
param['momentum'] = 0.9    #动量
param['weight_decay'] = 0. #权重衰减
param['checkpoint_dir'] = checkpoint_dir
param['disp_inter'] = 1 # 显示间隔
param['save_inter'] = 1 # 保存间隔
# 训练
best_model, model = dl.train_net(param, model, train_data, valid_data)


在obs中生成提交目录
import moxing as mox
mox.file.copy_parallel('/home/ma-user/work/RSC/RSC_Baseline/unet/', 'obs://obs-2020hwcc-baseline/submission/model/unet/')
mox.file.copy_parallel('/home/ma-user/work/RSC/ckpt/unet/checkpoint-best.pth', 'obs://obs-2020hwcc-baseline/submission/model/model_best.pth')
mox.file.copy_parallel('/home/ma-user/work/RSC/RSC_Baseline/subs/config.json', 'obs://obs-2020hwcc-baseline/submission/model/config.json')
mox.file.copy_parallel('/home/ma-user/work/RSC/RSC_Baseline/subs/customize_service.py', 'obs://obs-2020hwcc-baseline/submission/model/customize_service.py')
