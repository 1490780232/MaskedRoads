'''计算数据集的各项参数
'''
import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_std_mean(img_path):
    assert os.path.exists(img_path), '路径不存在{}'.format(img_path)
    files = os.listdir(img_path)
    imgs = []
    for img in tqdm(files, desc='STD MEAN'):
        img_array = np.array(Image.open(os.path.join(img_path, img)).convert('RGB'))
        w, h, c = img_array.shape
        imgs.append(img_array.reshape((w * h, c)))
    print('Computing...')
    imgs = np.concatenate(imgs)
    mean = list(np.mean(imgs, axis=0))
    std = list(np.std(imgs, axis=0))
    print('Mean: {}, Std: {}'.format(mean, std))
    with open('./output/img_std_mean.csv', 'w') as f:
        f.write(','.join([str(x) for x in mean]) + '\n')
        f.write(','.join([str(x) for x in std]) + '\n')


def class_weights(label_path):
    assert os.path.exists(label_path), '路径不存在{}'.format(label_path)
    files = os.listdir(label_path)
    n = 2
    pixel_count = np.array([0, 0], dtype=np.float64)
    for label_img in tqdm(files, desc='Label'):
        img_array = np.array(Image.open(os.path.join(label_path, label_img)))
        for c in range(n):
            pixel_count[c] += np.sum(img_array == c)
    pixel_percent = [str(x / sum(pixel_count)) for x in pixel_count]
    pixel_count = [str(x) for x in pixel_count]
    print(pixel_count, pixel_percent)
    with open('./class_weights.csv', 'w') as f:
        f.write("labels: "+"0   " + '1    \n')
        f.write("count: "+','.join(pixel_count) + '\n')
        f.write("percent: "+','.join(pixel_percent) + '\n')



if __name__ == "__main__":
    # if args.image_path is not None:
    #     compute_std_mean(args.image_path)
    class_weights("../data/train/labels/")