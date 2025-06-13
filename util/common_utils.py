import os

import torch
import random
import numpy as np
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid


def parse_yaml(path: str):
    yaml = YAML(typ='safe')
    with open(path, 'r') as f:
        args = yaml.load(f)
    return args


def set_all_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_all_dirs(hparams):
    base_dir = os.path.join(hparams['train']['save_dir'], hparams['train']['model_name'],
                            hparams['train']['task_name'])
    if not os.path.exists(os.path.join(base_dir, 'tensorboard')):
        os.makedirs(os.path.join(base_dir, 'tensorboard'))
    if not os.path.exists(os.path.join(base_dir, 'ckpt')):
        os.makedirs(os.path.join(base_dir, 'ckpt'))
    if not os.path.exists(os.path.join(base_dir, 'pic')):
        os.makedirs(os.path.join(base_dir, 'pic'))


def save_pics(hparams, source_img, target_img, output_img):
    base_path = os.path.join(hparams['train']['save_dir'], hparams['train']['model_name'],
                             hparams['train']['task_name'], 'pic')
    src = make_grid(source_img)
    tar = make_grid(target_img)
    out = make_grid(output_img)
    save_image(src, os.path.join(base_path, 'source.png'))
    save_image(tar, os.path.join(base_path, 'target.png'))
    save_image(out, os.path.join(base_path, 'output.png'))


def print_epoch_result(train_result, valid_result, epoch):
    print('Epoch:  {}'.format(epoch))
    for key, value in train_result.items():
        print(key, ':  ', value)
    if valid_result is not None:
        for key, value in valid_result.items():
            print(key, ':  ', value)


class MetricRecorder:
    def __init__(self):
        self.avg = .0
        self.count = 0
        self.value = .0
        self.total = .0

    def reset(self):
        self.avg = .0
        self.count = 0
        self.value = .0
        self.total = .0

    def update(self, value):
        value = round(value, 4)
        self.value = value
        self.total += value
        self.count += 1
        self.avg = round(self.total / self.count, 4)


class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_multi_scaler(self, scaler_dict, epoch):
        for key, value in scaler_dict.items():
            self.writer.add_scalar(key, value, epoch)
