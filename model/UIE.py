import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.color_net import Color_pred
from model.net import Net

class LocalExtremumConv(nn.Module):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        super(LocalExtremumConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv_min = nn.Conv2d(1, 1, kernel_size, stride, padding, dilation, bias=False).cuda()
        self.conv_max = nn.Conv2d(1, 1, kernel_size, stride, padding, dilation, bias=False).cuda()

        # 初始化卷积核权重
        self.init_weights()

    def init_weights(self):
        # 对于最小值卷积，我们希望权重是负的，且中心权重最小
        min_weights = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        min_weights *= -1.0 / (self.kernel_size ** 2)
        min_weights[0, 0, self.kernel_size // 2, self.kernel_size // 2] = -1.0
        self.conv_min.weight.data = min_weights.cuda()

        # 对于最大值卷积，我们希望权重是正的，且中心权重最大
        max_weights = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        max_weights *= 1.0 / (self.kernel_size ** 2)
        max_weights[0, 0, self.kernel_size // 2, self.kernel_size // 2] = 1.0
        self.conv_max.weight.data = max_weights.cuda()

    def forward(self, x):
        # 分割通道并应用卷积
        x_min = torch.min(x, dim=1, keepdim=True)[0]
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        # 应用最小值卷积
        dark_ch = F.relu(self.conv_min(x_min))

        # 应用最大值卷积
        bright_ch = F.relu(self.conv_max(x_max))

        return dark_ch, bright_ch

def get_illumination_channel(images, w):
    B, C, H, W = images.size()

    # 确保w是奇数，以便有一个中心像素
    if w % 2 == 0:
        w += 1

        # 计算填充大小
    pad_size = w // 2

    # 对图像进行填充
    padded_images = F.pad(images, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

    # 初始化LocalExtremumConv模块
    lec = LocalExtremumConv(w)

    # 计算暗通道和亮通道
    dark_ch, bright_ch = lec(padded_images)

    return dark_ch, bright_ch


def get_atmosphere(images, bright_ch, p=0.1):
    B, H, W = bright_ch.size()
    B, C, H, W = images.size()

    # 将图像和亮通道展平
    flat_images = images.view(B, -1, C)
    flat_bright_ch = bright_ch.view(B, -1)

    # 对每个图像单独处理
    atmospheres = []
    for b in range(B):
        # 对亮通道进行排序，并取前p个最小值的索引
        sorted_idx = torch.argsort(flat_bright_ch[b], descending=True)
        num_pixels = int(p * H * W)
        idx_to_consider = sorted_idx[:num_pixels]

        # 计算这些像素的平均颜色作为大气光
        atmosphere = torch.mean(flat_images[b, idx_to_consider, :], dim=0)
        atmospheres.append(atmosphere)

        # 将结果堆叠回批处理维度
    atmospheres = torch.stack(atmospheres, dim=0)
    return atmospheres

class CAT(nn.Module):
    def __init__(self, in_dim=3):
        super(CAT, self).__init__()

        self.color_net = Color_pred()
        self.net = Net()

        self.conv = nn.Conv2d(3, 4, kernel_size=1)

    def apply_color(self, image, color):

        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, color, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, x):

        B, C, H, W = x.shape
        img_high = self.net(x)
        color = self.color_net(x)

        # b = img_high.shape[0]
        # img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
        # img_high = torch.stack([self.apply_color(img_high[i, :, :, :], color[i, :, :]) for i in range(b)], dim=0)
        # img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)

        # img_high = img_high.permute(0, 2, 3, 1).reshape(B, -1, C)
        img_high = img_high * color
        # img_high = img_high.reshape(B, H, W, C).permute(0, 3, 1, 2)

        feat = self.conv(img_high)
        K, B = torch.split(feat, (1, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]

        dark, bright = get_illumination_channel(x, 3)
        A = get_atmosphere(x, bright.squeeze(dim=1))

        return x, A

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img = torch.Tensor(2, 3, 256, 256)
    net = CAT()
    high = net(img)
    print(high.shape)