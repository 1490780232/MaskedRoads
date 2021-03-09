# from yacs.config import CfgNode as CN
import sys
sys.path.append("E:/00_Doctor/00_project/2020_CCF_BDCI/HRNet-Semantic-Segmentation-pytorch-v1.1/lib/")
import yaml
from models.HRnet.seg_hrnet import get_seg_model
def get_config(path):
    f = open(path)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    # cfg = CN()
    # cfg.defrost()
    # cfg.merge_from_file("./lib/models/test.yaml")
    # cfg.freeze()
    # print(cfg.['MODEL'])
    # model = get_seg_model(cfg)
    return cfg
