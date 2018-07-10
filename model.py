import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init

from model_util import g_type2onehotclass

class Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, same_padding=True, stride=1, use_xavier=True, std_dev=1e-3,
                 activation_fn=True, bn=False, bn_decay=None, is_training=None):
        super(Conv2d, self).__init__()
        padding = int((kernel - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding)
        if use_xavier:
            init.xavier_normal(self.conv.weight.data)
            init.xavier_normal(self.conv.bias.data)
        else:
            init.normal_(self.conv.weight.data, std=std_dev)
            init.normal_(self.conv.bias.data, std=std_dev)
        bn_decay = 1 - bn_decay if bn_decay is not None else 0.1
        self.bn = nn.BatchNorm2d(out_channel, momentum=self.bn_decay) if bn else None
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        assert x.dim() == 4, "data should be BxCxHxW"
        x = self.conv(x)
        if self.bn is not None and self.training:
            x = self.bn(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class InstanceSeg(nn.Module):
    def __init__(self, num_points, bn_decay):
        super(InstanceSeg, self).__init__()
        self.conv1 = Conv2d(3, 64, kernel=1, same_padding=True, bn=True, bn_decay=bn_decay)
        self.conv2 = Conv2d(64, 64, kernel=1, same_padding=True, bn=True, bn_decay=bn_decay)
        self.conv3 = Conv2d(64, 64, kernel=1, same_padding=True, bn=True, bn_decay=bn_decay)
        self.conv4 = Conv2d(64, 128, kernel=1, same_padding=True, bn=True, bn_decay=bn_decay)
        self.conv5 = Conv2d(128, 1024, kernel=1, same_padding=True, bn=True, bn_decay=bn_decay)
        self.max_pool = nn.MaxPool2d(kernel_size=[num_points, 1])
        self.conv6 = Conv2d(64+1024+len(g_type2onehotclass.keys()), 512, kernel=1, same_padding=True,
                            bn=True, bn_decay=bn_decay)
        self.conv7 = Conv2d(512, 256, kernel=1, same_padding=True, bn=True, bn_decay=bn_decay)
        self.conv8 = Conv2d(256, 128, kernel=1, same_padding=True, bn=True, bn_decay=bn_decay)
        self.conv9 = Conv2d(128, 128, kernel=1, same_padding=True, bn=True, bn_decay=bn_decay)
        self.drop_out = nn.Dropout2d(p=0.5)
        self.conv10 = Conv2d(128, 2, kernel=1, same_padding=True, bn=True, activation_fn=False)

    def forward(self, point_cloud, one_hot_vec, end_points):
        assert point_cloud.dim() == 4, "input should be batch_size*channel_num*height*width"
        num_point = point_cloud.size()[2]
        conv = self.conv1(point_cloud)
        conv = self.conv2(conv)
        point_feat = self.conv3(conv)
        conv = self.conv4(point_feat)
        conv = self.conv5(conv)
        global_feat = self.max_pool(conv)
        one_hot_vec = one_hot_vec.reshape((-1, 3, 1, 1))
        global_feat = torch.cat((global_feat, one_hot_vec), dim=1)
        global_feat_expand = global_feat.repeat([1, 1, num_point, 1])
        concat_feat = torch.cat((point_feat, global_feat_expand), dim=1)
        conv = self.conv6(concat_feat)
        conv = self.conv7(conv)
        conv = self.conv8(conv)
        conv = self.conv9(conv)
        if self.training:
            conv = self.drop_out(conv)
        logits = self.conv10(conv)
        logits = logits.squeeze()
        return logits, end_points


class FrustumPointnets(nn.Module):
    def __init__(self):
        super(FrustumPointnets, self).__init__()

