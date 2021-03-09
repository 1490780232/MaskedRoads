from models import UNet
import utile.deeplearning as dl
import os
import numpy as np
import torch
from PIL import Image
from models.Deeplab_v3_plus.get_model import get_deeplabv3plus_resnet101
from models.Deeplab import DeepLabV3Res101, DeepLabV3Res50
from models.Deeplab_v3plus.generateNet import generate_Deeplab_v3
from models.Deeplab_v3plus.sync_batchnorm.replicate import patch_replication_callback
from models.HRnet.seg_hrnet import get_seg_model
from models.HRnet.get_cfg import get_config
from torch import nn
from utile import ext_transforms as et
if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = 1000000000000000
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "../../data/"
    train_imgs_dir = os.path.join(data_dir, "train/images/")
    val_imgs_dir = os.path.join(data_dir, "val/images/")
    train_labels_dir = os.path.join(data_dir, "train/labels/")
    val_labels_dir = os.path.join(data_dir, "val/labels/")
    train_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtRandomRotation(30),
        et.ExtRandomScale([0.5, 2]),
        et.ExtRandomCrop(size=(512, 512)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomVerticalFlip(),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    train_data = dl.RSCDataset(train_imgs_dir, train_labels_dir, transforms = train_transform)
    valid_data = dl.RSCDataset(val_imgs_dir, val_labels_dir, transforms = val_transform)
    checkpoint_dir = os.path.join("./ckpt/", 'deep_lab/')  # 模型保存路径
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    # model = UNet(3, 2).to(device)
    # model = DeepLabV3Res50().to(device)
    # model = generate_Deeplab_v3(2)
    # model = get_deeplabv3plus_resnet101(num_classes=2, output_stride=16, separable_conv=True)
    model = get_seg_model(get_config("./models/HRnet/config.yaml"))
    print(model)
    # model = nn.DataParallel(model)
    # patch_replication_callback(model)
    model.to(device)
    # 参数设置
    param = {}
    param['epochs'] = 41  # 训练轮数
    param['batch_size'] = 12  # 48 #4   # 批大小
    param['lr'] = 2e-2  # 学习率
    param['gamma'] = 0.9  # 学习率衰减系数
    param['step_size'] = 5  # 学习率衰减间隔
    param['momentum'] = 0.9  # 动量
    param['weight_decay'] = 0.  # 权重衰减
    param['checkpoint_dir'] = checkpoint_dir
    param['disp_inter'] = 1  # 显示间隔
    param['save_inter'] = 1  # 保存间隔
    param["loss_type"] = 'focal'
    param["warmup_factor"] = 0.01
    param["warmup_epochs"] = 5  # warm up epochs
    param["warmup_method"] = "linear"  #option: 'linear','constant'
    param['weights'] = None
    #1 - x for x in [0.08658596610788535, 0.9134140338921146]]
    # 训练
    best_model, model = dl.train_net(param, model, train_data, valid_data)