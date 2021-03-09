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

Image.MAX_IMAGE_PIXELS = 1000000000000000
from models.Deeplab import DeepLabV3Res101, DeepLabV3Res50
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from tools.utils import setup_logger
from solver.lr_scheduler import WarmupMultiStepLR

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import cv2
# import segmentation_refinement as refine
# import matplotlib.pyplot as plt
from losses import FocalLoss2d, mIoULoss2d, CrossEntropyLoss2d


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
class RSCDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transforms):
        self.transforms = transforms
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        try:
            img_trans = img_nd.transpose(2, 0, 1)
        except:
            print(img_nd.shape)
        if img_trans.max() > 1: img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.transforms(img, mask)
        # mask = np.array(mask)
        return {
            'trace': img,
            'label': mask
        }
        # img = self.preprocess(img)
        # mask = np.array(mask)
        # return {
        #     'trace': img.astype(np.float32),
        #     'label': mask.astype(np.long)
        # }


def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def train_net(param, model, train_data, valid_data, plot=True):
    logger = setup_logger('{}'.format("trainning deeplabv3-plus"), "./output/")
    logger.info('start training')
    # 初始化参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']
    gamma = param['gamma']
    step_size = param['step_size']
    momentum = param['momentum']
    weight_decay = param['weight_decay']
    disp_inter = param['disp_inter']
    save_inter = param['save_inter']
    checkpoint_dir = param['checkpoint_dir']
    loss_type = param["loss_type"]
    warmup_factor = param["warmup_factor"]
    warmup_epochs = param["warmup_epochs"]
    warmup_method = param["warmup_method"]
    weights = param['weights']
    # print(weights)
    loss_map = {
        'focal': FocalLoss2d,
        'ce': CrossEntropyLoss2d,
        'miou': mIoULoss2d
    }
    train_size = train_data.__len__()
    valid_size = valid_data.__len__()
    # c, y, x = train_data.__getitem__(0)['trace'].shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=8)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # scheduler = WarmupMultiStepLR(optimizer, [5,10,15,20,25,30,35,40], gamma,
    #                             warmup_factor,
    #                             warmup_epochs, warmup_method)
    # criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
    criterion = loss_map[loss_type](weight=weights).to(device)
    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_loss = 1e50
    best_miou = 0
    best_mode = copy.deepcopy(model)
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss_per_epoch = 0
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['trace'], batch_samples['label']
            # data, target = Variable(data.to(device)), Variable(target.to(device))
            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)
            # print(data.shape, target.shape)
            optimizer.zero_grad()
            pred = model(data)
            target = target.long()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss_per_epoch += loss.item()
        # 验证阶段
        model.eval()
        valid_loss_per_epoch = 0
        miou_per_epoch = 0
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target = batch_samples['trace'], batch_samples['label']
                data = data.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.long)
                # data, target = Variable(data.to(device)), Variable(target.to(device))
                pred = model(data)
                loss = criterion(pred, target)
                pred = pred.cpu().data.numpy()
                pred = np.argmax(pred, axis=1)
                miou, _, _ = get_metrics(pred, target.cpu().data.numpy())
                valid_loss_per_epoch += loss.item()
                miou_per_epoch += miou
        train_loss_per_epoch = train_loss_per_epoch / train_size
        valid_loss_per_epoch = valid_loss_per_epoch / valid_size
        miou_per_epoch = miou_per_epoch / valid_size
        train_loss_total_epochs.append(train_loss_per_epoch)
        valid_loss_total_epochs.append(valid_loss_per_epoch)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        if epoch % save_inter == 0:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
        # 保存最优模型
        if miou_per_epoch > best_miou:  # valid_loss_per_epoch < best_loss : # train_loss_per_epoch valid_loss_per_epoch and miou_per_epoch> best_miou
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(checkpoint_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_loss = valid_loss_per_epoch
            best_miou = miou_per_epoch
            best_mode = copy.deepcopy(model)
            logger.info('Best Epoch:{}, Training Loss:{:.8f}, Validation Loss:{:.8f}, miou:{:.8f}:'.format(epoch,
                                                                                                           train_loss_per_epoch,
                                                                                                           valid_loss_per_epoch,
                                                                                                           miou_per_epoch))
        scheduler.step()
        # 显示loss
        if epoch % disp_inter == 0:
            print('Epoch:{}, Training Loss:{:.8f}, Validation Loss:{:.8f}, miou:{:.8f}:'.format(epoch,
                                                                                                train_loss_per_epoch,
                                                                                                valid_loss_per_epoch,
                                                                                                miou_per_epoch))
            logger.info('Epoch:{}, Training Loss:{:.8f}, Validation Loss:{:.8f}, miou:{:.8f}:'.format(epoch,
                                                                                                      train_loss_per_epoch,
                                                                                                      valid_loss_per_epoch,
                                                                                                      miou_per_epoch))
    # 训练loss曲线
    # if plot:
    #     x = [i for i in range(epochs)]
    #     fig = plt.figure(figsize=(12, 4))
    #     ax = fig.add_subplot(1, 2, 1)
    #     ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='训练集loss')
    #     ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='验证集loss')
    #     ax.set_xlabel('Epoch', fontsize=15)
    #     ax.set_ylabel('CrossEntropy', fontsize=15)
    #     ax.set_title(f'训练曲线', fontsize=15)
    #     ax.grid(True)
    #     plt.legend(loc='upper right', fontsize=15)
    #     ax = fig.add_subplot(1, 2, 2)
    #     ax.plot(x, epoch_lr,  label='Learning Rate')
    #     ax.set_xlabel('Epoch', fontsize=15)
    #     ax.set_ylabel('Learning Rate', fontsize=15)
    #     ax.set_title(f'学习率变化曲线', fontsize=15)
    #     ax.grid(True)
    #     plt.legend(loc='upper right', fontsize=15)
    #     plt.show()

    return best_mode, model


def pred(model, data):
    target_l = 1024
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.transpose(2, 0, 1)
    if data.max() > 1: data = data / 255
    c, x, y = data.shape
    label = np.zeros((x, y))
    x_num = (x // target_l + 1) if x % target_l else x // target_l
    y_num = (y // target_l + 1) if y % target_l else y // target_l
    for i in tqdm(range(x_num)):
        for j in range(y_num):
            x_s, x_e = i * target_l, (i + 1) * target_l
            y_s, y_e = j * target_l, (j + 1) * target_l
            img = data[:, x_s:x_e, y_s:y_e]
            img = img[np.newaxis, :, :, :].astype(np.float32)
            img = torch.from_numpy(img)
            img = Variable(img.to(device))
            out_l = model(img)
            out_l = out_l.cpu().data.numpy()
            out_l = np.argmax(out_l, axis=1)[0]
            label[x_s:x_e, y_s:y_e] = out_l.astype(np.int8)
    return label


def get_metrics(gd_array: np.ndarray, pred_array: np.ndarray) -> (float, list, float):
    '''计算图片的平均IoU

    Args:
        gd_array(np.ndarray): Ground Truth
        pred_array(np.ndarray): 预测数据

    Return:
        mIoU(float): 平均IoU
        IoUs(list): 每一类的IoU
        mpa(float): 平均每类的像素正确率
    '''
    num_classes = 2
    IoUs = []
    assert pred_array.shape == gd_array.shape, '预测图片({})和GD({})大小不一致'.format(pred_array.shape, gd_array.shape)
    for c in range(0, num_classes):
        gmask = gd_array == c
        pmask = pred_array == c
        iarea = np.sum(gmask & pmask)
        uarea = np.sum(gmask | pmask)
        IoU = iarea / uarea
        IoUs.append(IoU)
    mIoU = np.nanmean(IoUs)
    acc = np.sum(gd_array == pred_array) / pred_array.size
    return mIoU, IoUs, acc


def cal_metrics(pred_label, gt):
    def _generate_matrix(gt_image, pre_image, num_class=2):
        mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask
        label = num_class * gt_image[mask].astype('int') + pre_image[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class ** 2)
        confusion_matrix = count.reshape(num_class, num_class)  # 21 * 21(for pascal)
        return confusion_matrix

    def _Class_IOU(confusion_matrix):
        MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        return MIoU

    confusion_matrix = _generate_matrix(gt.astype(np.int8), pred_label.astype(np.int8))
    miou = _Class_IOU(confusion_matrix)
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    return miou, acc

#
# def predict():
#     device = "cuda"
#     # model = UNet(3, 2).to(device)
#     # model = DeepLabV3Res50().to(device)
#     model = get_deeplabv3plus_resnet101(num_classes=2, output_stride=16, separable_conv=True).to(device)
#     model.load_state_dict(torch.load("../ckpt/deep_lab_v3_plus/checkpoint-best_0_weight.pth")['state_dict'])
#     mious, ious, accs = [], [], []
#     root = "../../../data/val/"
#     # root = "../../../data/val/labels/"
#     files = os.listdir(root + "images/")
#     for file in tqdm(files):
#         data = Image.open(os.path.join(root, "images", file))
#         data = np.array(data)  # (array([0.46375846, 0.90893313]), 0.9155817031860352)
#         img = pred(model, data)
#         label = Image.open(os.path.join(root, "labels", file))
#         label = np.array(label)
#         # refiner = refine.Refiner(device="cuda:0")
#         # output = refiner.refine(cv2.imread("../data/val/images/182_7_53.png"), cv2.imread("../data/val/labels/182_7_53.png", cv2.IMREAD_GRAYSCALE), fast = False, L = 900)
#         # print(np.unique(output))
#         miou, iou, acc = get_metrics(img, label)
#         # print(iou)
#         mious.append(miou)
#         ious.append(iou)
#         accs.append(acc)
#         # print(get_metrics(img,label))#cal_metrics(img,label))
#         # print(cal_metrics(img,label))
#     print(np.average(mious), np.nanmean(ious, axis=0), np.average(accs))


def predict():
    device = "cuda"
    # model = UNet(3, 2).to(device)
    # model = DeepLabV3Res50().to(device)
    model = get_deeplabv3plus_resnet101(num_classes=2, output_stride=16, separable_conv=True).to(device)
    model.load_state_dict(torch.load("./model_best.pth")['state_dict'])
    mious, ious, accs = [], [], []
    root = "../../../data/val/"
    # root = "../../../data/val/labels/"
    files = os.listdir(root + "images/")
    for file in tqdm(files):
        data = Image.open(os.path.join(root, "images", file))
        data = np.array(data)  # (array([0.46375846, 0.90893313]), 0.9155817031860352)
        img = pred(model, data)
        label = Image.open(os.path.join(root, "labels", file))
        label = np.array(label)
        # refiner = refine.Refiner(device="cuda:0")
        # output = refiner.refine(cv2.imread("../data/val/images/182_7_53.png"), cv2.imread("../data/val/labels/182_7_53.png", cv2.IMREAD_GRAYSCALE), fast = False, L = 900)
        # print(np.unique(output))
        miou, iou, acc = get_metrics(img, label)
        # print(iou)
        mious.append(miou)
        ious.append(iou)
        accs.append(acc)
        # print(get_metrics(img,label))#cal_metrics(img,label))
        # print(cal_metrics(img,label))
    print(np.average(mious), np.nanmean(ious, axis=0), np.average(accs))

if __name__ == "__main__":
    # device = "cuda"
    # model = get_deeplabv3plus_resnet101(num_classes=2, output_stride=16, separable_conv=True).to(device)
    # model.load_state_dict(torch.load("../ckpt/deep_lab/checkpoint-best.pth")['state_dict'])
    # data = Image.open("../../../data/val/images/182_6_9.png")
    # data = np.array(data)  # (array([0.46375846, 0.90893313]), 0.9155817031860352)
    # img = pred(model, data)
    # # print(img.shape)
    # # img = np.argmax(img, axis=1)[0]
    # label = Image.open("../../../data/val/labels/182_6_9.png")
    # label = np.array(label)
    # Image.fromarray(img.astype(np.uint8)*200).show()
    # mious, ious, accs = [], [], []
    predict()
    # im= Image.open("../../../data/val/labels/382_17_24.png")
    # im =np.array(im)*255
    # # print(im)
    # im = Image.fromarray(im.astype(np.uint8))
    # im.show()