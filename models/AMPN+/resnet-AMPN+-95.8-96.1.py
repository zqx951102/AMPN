from collections import OrderedDict
#resnet-Best-Seresnet

import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import torch.nn as nn
import torch.fft
import torchvision as tv
import torchvision
from torchvision import datasets, models, transforms
from torch.nn.modules.utils import _pair
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
import torch.nn as nn
import torch.nn.functional as F

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
#backbone seresnet50
import timm
import pdb


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()

        # 1x1 convolution
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # 3x3 convolution
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

        # 5x5 convolution
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=in_channels)

        # 7x7 convolution
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, groups=in_channels)

        # pointwise convolution
        self.pw_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

        # batch normalization
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 1x1 convolution
        x1 = self.conv1x1(x)
        x1 = F.relu(x1)

        # 3x3 convolution
        x3 = self.conv3x3(x)
        x3 = F.relu(x3)

        # 5x5 convolution
        x5 = self.conv5x5(x)
        x5 = F.relu(x5)

        # 7x7 convolution
        x7 = self.conv7x7(x)
        x7 = F.relu(x7)

        # concatenate feature maps
        x_cat = torch.cat((x1, x3, x5, x7), dim=1)

        # pointwise convolution
        out = self.pw_conv(x_cat)

        # batch normalization
        out = self.bn(out)

        # relu activation
        out = F.relu(out)

        return out


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


# gauss
def GaussProjection(x, mean, std):
    sigma = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2)) / sigma
    return x_out


class BasicConv(nn.Module):  
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# channel attention
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg','max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(  # mlp
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),  # 16
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':  
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':  
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # Gauss modulation
        mean = torch.mean(channel_att_sum).detach()
        std = torch.std(channel_att_sum).detach()
        scale = GaussProjection(channel_att_sum, mean, std).unsqueeze(2).unsqueeze(3).expand_as(x)

        # scale = scale / torch.max(scale)
        return x * scale


class ChannelPool(nn.Module):  
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.pool = ChannelPool()  
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)  

    def forward(self, x):
        x_pool = self.pool(x)  
        x_out = self.spatial(x_pool)  

        # Gauss modulation
        mean = torch.mean(x_out).detach()
        std = torch.std(x_out).detach()
        scale = GaussProjection(x_out, mean, std) 

        # scale = scale / torch.max(scale)
        return x * scale  


class TokenMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_att = SpatialGate() 
        self.multi_conv = MultiScaleConv(256, 256)

    def SpatialGate_forward(self, x): 
        residual = x  
        x = self.spatial_att(x)
        x = residual + x
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x_pre_norm = self.SpatialGate_forward(x)
        x = self.multi_conv(x_pre_norm)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, reduction_ratio=16, pool_types=['avg','max']):
        super().__init__()
        self.channel_att = ChannelGate(num_features, reduction_ratio, pool_types)
        self.multi_conv = MultiScaleConv(256, 256)

    def ChannelGate_forward(self, x):
        residual = x
        x = self.channel_att(x)
        x = residual + x
        return x

    def forward(self, x):
        residual = x
        x_pre_norm = self.ChannelGate_forward(x)
        x = self.multi_conv(x_pre_norm)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, reduction_ratio=16, pool_types=['avg','max']):
        super().__init__()
        self.token_mixer = TokenMixer()
        self.channel_mixer = ChannelMixer(num_features, reduction_ratio, pool_types)

    def forward(self, x): 
        x = self.token_mixer(x)
        x = self.channel_mixer(x)

        return x


class MGHA_Mixer(nn.Module):
    def __init__(
            self,
            in_channels=256,
            num_features=256,
            reduction_ratio=16,
            pool_types=['avg','max']
    ):
        super().__init__()

        self.mixers = MixerLayer(num_features, reduction_ratio, pool_types)
        self.simam = SimAM() 

    def forward(self, x):
        residual = x
        embedding = self.mixers(x)
        embedding_final = embedding + self.simam(x) + residual  
        return embedding_final



class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        self.feature1 = nn.Sequential(resnet.conv1,
                                      resnet.bn1, resnet.act1, resnet.maxpool)
        self.layer1 = nn.Sequential(resnet.layer1)
        self.layer2 = nn.Sequential(resnet.layer2)
        self.layer3 = nn.Sequential(resnet.layer3)

        self.out_channels = 1024

    def forward(self, x):
        feat = self.feature1(x)
        layer1 = self.layer1(feat)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        return OrderedDict([["feat_res4", layer3]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()
        self.layer4 = nn.Sequential(resnet.layer4)  # res5
        self.out_channels = [1024, 2048]

        self.MGHA_model = MGHA_Mixer(256)  
        self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        self.qconv2 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1)

    def forward(self, x):
        qconv1 = self.qconv1(x)  
        x_sc_mlp_feat = self.MGHA_model(qconv1)  
        qconv2 = self.qconv2(x_sc_mlp_feat) 

        layer5_feat = self.layer4(qconv2)

        x_feat = F.adaptive_max_pool2d(qconv2, 1)

        feat = F.adaptive_max_pool2d(layer5_feat, 1)

        return OrderedDict([["feat_res4", x_feat], ["feat_res5", feat]])


def build_resnet(name="seresnet50", pretrained=True):
    resnet = timm.create_model(name, pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)
# def build_resnet(name="resnet50", pretrained=True):
#     from torchvision.models import resnet
#     resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
#     resnet_model = resnet.resnet50(pretrained=True)
#
#     # freeze layers
#     resnet_model.conv1.weight.requires_grad_(False)
#     resnet_model.bn1.weight.requires_grad_(False)
#     resnet_model.bn1.bias.requires_grad_(False)
#
#     return Backbone(resnet_model), Res5Head(
#         resnet_model)  
