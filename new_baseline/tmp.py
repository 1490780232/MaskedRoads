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
from torch import nn
from utile import ext_transforms as et
from torchvision.transforms.functional import normalize
class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)
train_transform = et.ExtCompose([
    et.ExtResize( 512 ),
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

data_dir = "../../data/"
train_imgs_dir = os.path.join(data_dir, "train/images/")
val_imgs_dir = os.path.join(data_dir, "val/images/")
train_labels_dir = os.path.join(data_dir, "train/labels/")
val_labels_dir = os.path.join(data_dir, "val/labels/")
train_data = dl.RSCDataset(train_imgs_dir, train_labels_dir, transforms=train_transform)
valid_data = dl.RSCDataset(val_imgs_dir, val_labels_dir, transforms=val_transform)
data = train_data.__getitem__(10)
image = data['trace']
label = data['label']
print(image.shape)
image = image.detach().cpu().numpy()
label = label.detach().cpu().numpy()
denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
label = (label*255).astype(np.uint8)
# target = loader.dataset.decode_target(target).astype(np.uint8)
# pred = loader.dataset.decode_target(pred).astype(np.uint8)
Image.fromarray(image).show()
Image.fromarray(label).show()

# Image.fromarray(target).save('results/%d_target.png' % img_id)
# Image.fromarray(pred).save('results/%d_pred.png' % img_id)
