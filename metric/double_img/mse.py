import torch
from torchvision.transforms import ToTensor


def calc_mse(tar_img, prd_img):
    tar_img = ToTensor()(tar_img)
    prd_img = ToTensor()(prd_img)
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    return rmse.item()
