# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from .resnet import ResNet
from maskrcnn_benchmark.layers.triplet_loss import triplet_loss
from maskrcnn_benchmark.config import cfg
import torch.nn.functional as F
from IPython import embed

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class REID_Baseline(nn.Module):
    def __init__(self, cfg):
        super(REID_Baseline, self).__init__()
        self.cfg = cfg.clone()
        self.base = ResNet(cfg.REID.MODEL.LAST_STRIDE)
        self.base.load_param(cfg.REID.MODEL.PRETRAIN_PATH)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = cfg.REID.DATASETS.ID_NUM
        # if cfg.MODEL.MASK_ON:
        #     self.in_planes = 4096
        # else:
        #     self.in_planes = 2048
        self.in_planes = 2048
        self.reid_feat_dim = 256

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.queue_size = 5000

        self.margin = self.cfg.TRIPLET.MARGIN
        self.lut_momentum = 0.0  ###
        self.reid_feat_dim = self.in_planes

        # self.register_buffer('lut', torch.zeros(self.num_pid, self.reid_feat_dim).cuda())
        self.register_buffer('lut1', torch.zeros(self.num_classes, self.reid_feat_dim).cuda())
        self.register_buffer('lut2', torch.zeros(self.num_classes, self.reid_feat_dim).cuda())
        self.register_buffer('queue', torch.zeros(self.queue_size, self.reid_feat_dim).cuda())

    def forward(self, img_list, pid_label, iteration, mode):
        feats = []
        feat = 0
        for x in img_list:
            global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            feats.append(self.bottleneck(global_feat))  # normalize for angular softmax
        for i in feats:
            feat += 1 / len(feats) * i
        if mode == 'test' or mode == 'query':
            return feat
        losses = {}
        labels = pid_label.type(torch.cuda.LongTensor)
        if 'softmax' in cfg.REID.DATALOADER.SAMPLER:
            cls_score = self.classifier(feat)
            cls_loss = F.cross_entropy(cls_score, labels)
            losses.update(dict(cls_loss=cls_loss))
        if 'triplet' in cfg.REID.DATALOADER.SAMPLER:
            tri_loss = triplet_loss(global_feat, pid_label, self.lut1, self.lut2, self.queue, self.lut_momentum,
                                    iteration, self.margin)
            losses.update(dict(tri_loss=tri_loss))
        return losses

















