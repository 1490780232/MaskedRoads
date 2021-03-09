import os
import copy
import torch
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import random
# import moxing as mox
from models import UNet
import logging
from glob import glob
from PIL import Image

from models.Deeplab_v3_plus.get_model import get_deeplabv3plus_resnet101

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_deeplabv3plus_resnet101(num_classes=2, output_stride=16, separable_conv=True).to(device)
model.load_state_dict(torch.load("../ckpt/deep_lab_v3_plus/model_best.pth")['state_dict'])
state = {'epoch': 1, 'state_dict': model.state_dict()}
torch.save(state, 'model_best.pth', _use_new_zipfile_serialization=False)