import torch
from torchvision.transforms import ToTensor


def calc_psnr(tar_img, prd_img):
    tar_img = ToTensor()(tar_img)
    prd_img = ToTensor()(prd_img)
    imdff = torch.clamp(prd_img, 0, 1) - torch.clamp(tar_img, 0, 1)
    rmse = (imdff ** 2).mean().sqrt()
    ps = 20 * torch.log10(1 / rmse)
    return ps.item()
