import os
import torch.cuda
from torch import no_grad, load
from model.UIE import CAT
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms import Pad, Resize


def predict_one_dataset(model, img_dir, save_dir, ckpt_name, use_resize, pad_scale):
    ckpt_info = load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt_info['model'])
    img_names = os.listdir(img_dir)
    for img_name in img_names:
        torch.cuda.empty_cache()
        model = model.cuda()
        img_path = os.path.join(img_dir, img_name)
        source_img = (read_image(img_path) / 255.0)
        img_h = source_img.shape[1]
        img_w = source_img.shape[2]
        #pad_h = (pad_scale - img_h % pad_scale) % pad_scale
        #pad_w = (pad_scale - img_w % pad_scale) % pad_scale
        #source_img = Pad(padding=[pad_w, pad_h, 0, 0], padding_mode='reflect')(source_img)
        source_img = source_img.unsqueeze(0).cuda()
        if use_resize:
            source_img = Resize((256, 256))(source_img)
        with no_grad():
            output_img , a = model(source_img)
            output_img = output_img.clamp(0, 1).cpu()
        save_image(output_img, os.path.join(save_dir, img_name))


model = CAT()
img_dir = "/workspace/Lianghui/dataset/EUVP1/test/input/"
save_dir = "/workspace/Lianghui/UIEB-SSC-filter6+Mul-enhance+L1/test-EUVP_in-UIEB/"
ckpt_path = "./log/UIENet/base/ckpt/best_psnr.pth"
predict_one_dataset(model, img_dir, save_dir, ckpt_path, True, 0)
