from pytorch_msssim import ssim
from torchvision.transforms import ToTensor


def calc_ssim(tar_img, prd_img):
    tar_img = ToTensor()(tar_img).unsqueeze(0)
    prd_img = ToTensor()(prd_img).unsqueeze(0)
    return ssim(tar_img, prd_img, data_range=1.0, size_average=True).item()
