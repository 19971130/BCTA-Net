import os

from util.common_utils import parse_yaml
from train import train
from model.UIE import CAT

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] ='2'
    args = parse_yaml('./config.yaml')
    model = CAT().cuda()
    train(model, args)
