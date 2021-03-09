# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
from models.Deeplab_v3plus.deeplabv3plus import deeplabv3plus
# from net.supernet import SuperNet
# from net.EANet import EANet
# from net.DANet import DANet
# from net.deeplabv3plushd import deeplabv3plushd
# from net.DANethd import DANethd


# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------
import torch
import argparse
import os
import sys
import cv2
import time


class Configuration():
	def __init__(self, MODEL_NUM_CLASSES):
		self.MODEL_NAME = 'deeplabv3plus'
		self.MODEL_BACKBONE = 'res101_atrous'
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = MODEL_NUM_CLASSES

		self.TRAIN_BN_MOM = 0.0003
		# self.TRAIN_EPOCHS = 46

def generate_Deeplab_v3(num_classes):
	cfg = Configuration(MODEL_NUM_CLASSES=num_classes)
	if cfg.MODEL_NAME == 'deeplabv3plus' or cfg.MODEL_NAME == 'deeplabv3+':
		return deeplabv3plus(cfg)
	# if cfg.MODEL_NAME == 'supernet' or cfg.MODEL_NAME == 'SuperNet':
	# 	return SuperNet(cfg)
	# if cfg.MODEL_NAME == 'eanet' or cfg.MODEL_NAME == 'EANet':
	# 	return EANet(cfg)
	# if cfg.MODEL_NAME == 'danet' or cfg.MODEL_NAME == 'DANet':
	# 	return DANet(cfg)
	# if cfg.MODEL_NAME == 'deeplabv3plushd' or cfg.MODEL_NAME == 'deeplabv3+hd':
	# 	return deeplabv3plushd(cfg)
	# if cfg.MODEL_NAME == 'danethd' or cfg.MODEL_NAME == 'DANethd':
	# 	return DANethd(cfg)
	else:
		raise ValueError('generateNet.py: network %s is not support yet'%cfg.MODEL_NAME)
