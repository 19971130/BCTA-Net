import os
import torch
import math
import torch.nn as nn
from einops import rearrange
from torch.nn import init
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x.size() 30,40,50,30
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 30,1,50,30
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 30,1,50,30
        return self.sigmoid(x)  # 30,1,50,30


class down_Block(nn.Module):
    def __init__(self, in_channels, stride=2):

        kernel_size = 3

        super(down_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv2d(in_channels, stride*in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out

class up_Block(nn.Module):
    def __init__(self, in_channels, out_channels):

        kernel_size = 4

        super(up_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=1, stride=2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

    def forward(self, x):
        return self.mlp(x)

class Attention(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.heads = 1
        self.qk_dim = dim
        self.qk_scale = self.qk_dim ** -0.5

        self.act = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

        self.qkv_linear = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, C, H, W = x.shape

        q, k, v = self.qkv_linear(x.permute(0, 2, 3, 1)).chunk(3, dim=-1)
        q = q.reshape(B, -1, C).permute(0, 2, 1)
        k = k.reshape(B, -1, C)
        v = v.reshape(B, -1, C).permute(0, 2, 1)
        # q = rearrange(q, 'b h w (m c) -> b m c n', m=self.heads)
        #
        # k = rearrange(k, 'b h w (m c) -> b m n c', m=self.heads)
        #
        # v = rearrange(v, 'b h w (m c) -> b m c n', m=self.heads)

        attn = q @ k * self.qk_scale
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1)

        out = self.proj(out)
        out = self.proj_drop(out)

        attn_out = out.permute(0, 1, 2).reshape(B, C, H, W)

        conv_out = self.conv(x)
        out = attn_out + conv_out

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):

    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, in_chn - lda_out_channels * 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # initialize
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # the same effect as lda operation in the paper, but save much more memory
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        x = self.layer2(x)
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = self.layer4(x)
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))

        vec = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)

        out = {}
        out['hyper_in_feat'] = x
        out['target_in_vec'] = vec

        return out

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def resnet50_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model

class HyperNet(nn.Module):
    """
    Hyper network for learning perceptual rules.

    Args:
        lda_out_channels: local distortion aware module output size.
        hyper_in_channels: input feature channels for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        feature_size: input feature map width/height for hyper network.

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.

    """
    def __init__(self, lda_out_channels, hyper_in_channels, target_in_size, target_fc1_size, target_fc2_size, target_fc3_size, target_fc4_size, feature_size):
        super(HyperNet, self).__init__()

        self.hyperInChn = hyper_in_channels
        self.target_in_size = target_in_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.f3 = target_fc3_size
        self.f4 = target_fc4_size
        self.feature_size = feature_size

        self.res = resnet50_backbone(lda_out_channels, target_in_size, pretrained=True)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Conv layers for resnet output features
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True)
        )

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn, int(self.target_in_size * self.f1 / feature_size ** 2), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn, int(self.f1 * self.f2 / feature_size ** 2), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn, self.f2)

        self.fc3w_conv = nn.Conv2d(self.hyperInChn, int(self.f2 * self.f3 / feature_size ** 2), 3, padding=(1, 1))
        self.fc3b_fc = nn.Linear(self.hyperInChn, self.f3)

        self.fc4w_conv = nn.Conv2d(self.hyperInChn, int(self.f3 * self.f4 / feature_size ** 2), 3, padding=(1, 1))
        self.fc4b_fc = nn.Linear(self.hyperInChn, self.f4)

        self.fc5w_fc = nn.Linear(self.hyperInChn, self.f4)
        self.fc5b_fc = nn.Linear(self.hyperInChn, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 2:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

    def forward(self, img):
        feature_size = self.feature_size

        res_out = self.res(img)

        # input vector for target net
        target_in_vec = res_out['target_in_vec'].view(-1, self.target_in_size, 1, 1)

        # input features for hyper net
        hyper_in_feat = self.conv1(res_out['hyper_in_feat']).view(-1, self.hyperInChn, feature_size, feature_size)

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size, 1, 1)
        target_fc1b = self.fc1b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_conv(hyper_in_feat).view(-1, self.f3, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f3)

        target_fc4w = self.fc4w_conv(hyper_in_feat).view(-1, self.f4, self.f3, 1, 1)
        target_fc4b = self.fc4b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, self.f4)

        target_fc5w = self.fc5w_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1, self.f4, 1, 1)
        target_fc5b = self.fc5b_fc(self.pool(hyper_in_feat).squeeze()).view(-1, 1)

        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b
        out['target_fc4w'] = target_fc4w
        out['target_fc4b'] = target_fc4b
        out['target_fc5w'] = target_fc5w
        out['target_fc5b'] = target_fc5b

        return out

class Former(nn.Module):
    def __init__(self, dim, size, mlp_ratio=4., norm_layer=nn.InstanceNorm2d):
        super().__init__()

        self.ca = CALayer(channel=dim)
        self.sa = SpatialAttention()

        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = norm_layer(dim)

        self.attn = Attention(dim)

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

        self.blk = Bottleneck(inplanes=dim, planes=dim)
        # self.hyper = HyperNet(dim, dim, 224, 112, 56, 28, 14, 7)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):

        res = x
        ca = self.ca(x)
        sa = self.sa(x)

        x = self.pos_embed(x)
        x = x + self.attn(self.norm(x))
        x = self.mlp(x + ca + sa) + res

        # x = x + ca + sa

        # weight = self.conv(x)

        # x = weight * x
        # x = self.mlp(x) + res
        # x = self.blk(x) * self.conv(x)

        return x

class Net(nn.Module):

    def __init__(self, in_dim=3):
        super(Net, self).__init__()

        self.down1 = down_Block(in_channels=32)
        self.down2 = down_Block(in_channels=64)
        self.down3 = down_Block(in_channels=128, stride=1)

        self.up1 = up_Block(in_channels=128, out_channels=64)
        self.up2 = up_Block(in_channels=64, out_channels=32)
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1)
        )

        self.res1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1)
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1)
        )

        self.Former1 = nn.Sequential(
            Former(dim=32, size=128),
            Former(dim=32, size=128),
        )
        self.Former2 = nn.Sequential(
            Former(dim=64, size=64),
            Former(dim=64, size=64),
        )
        self.Former3 = nn.Sequential(
            Former(dim=128, size=32),
            Former(dim=128, size=32),
        )

        self.Former4 = nn.Sequential(
            Former(dim=128, size=32),
            Former(dim=128, size=32),
        )
        self.Former5 = nn.Sequential(
            Former(dim=64, size=64),
            Former(dim=64, size=64),
        )
        self.Former6 = nn.Sequential(
            Former(dim=32, size=128),
            Former(dim=32, size=128),
        )

    def forward(self, img_low):


        img_128 = F.interpolate(img_low, scale_factor=0.5, mode='bilinear', align_corners=False)
        img_64 = F.interpolate(img_low, scale_factor=0.25, mode='bilinear', align_corners=False)

        layer = self.conv(img_low)

        layer1 = self.down1(self.Former1(layer))

        x2 = self.Former2(layer1 + self.conv1(img_128))
        layer2 = self.down2(x2)

        x3 = self.Former3(layer2 + self.conv2(img_64))
        layer3 = self.down3(x3)

        layer4 = self.up1(self.Former4(layer3))
        layer5 = self.up2(self.Former5(layer4) + self.res2(x2))
        layer6 = self.up3(self.Former6(layer5) + self.res1(layer))

        out = self.conv_out(layer6)

        return out

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES']='0'
    img = torch.Tensor(2, 3, 256, 256)
    net = Net()
    high = net(img)
    print(high.shape)