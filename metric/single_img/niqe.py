import cv2
import pyiqa
import torch
from pyiqa import *
from torchvision.transforms import ToTensor

# input_path = r'D:\07zr\UIEB\test\target\(5).png'
niqe = pyiqa.create_metric('niqe', device=torch.device('cuda'))

def calc_niqe(prd_img):

    # prd_img = cv2.imread(prd_img)
    prd_img = ToTensor()(prd_img)

    return niqe(prd_img.unsqueeze(dim=0)).item()

# if __name__ == '__main__':
#
#     print(calc_niqe(input_path))
