import numpy as np
import logging
import os
import sys

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_metrics(gd_array:np.ndarray, pred_array:np.ndarray)->(float, list, float):
    '''计算图片的平均IoU

    Args:
        gd_array(np.ndarray): Ground Truth
        pred_array(np.ndarray): 预测数据

    Return:
        mIoU(float): 平均IoU
        IoUs(list): 每一类的IoU
        mpa(float): 平均每类的像素正确率
    '''
    num_classes = 4
    IoUs = []
    assert pred_array.shape == gd_array.shape, '预测图片({})和GD({})大小不一致'.format(pred_array.shape, gd_array.shape)
    for c in range(1, num_classes):
        gmask = gd_array == c
        pmask = pred_array == c
        iarea = np.sum(gmask & pmask)
        uarea = np.sum(gmask | pmask)
        IoU = iarea / uarea
        IoUs.append(IoU)
    mIoU = np.nanmean(IoUs)
    acc = np.sum(gd_array == pred_array) / pred_array.size
    return mIoU, IoUs, acc