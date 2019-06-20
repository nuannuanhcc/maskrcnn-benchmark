from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):
    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.lut.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(inputs, targets):
            # from IPython import embed
            # embed()
            self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
            self.lut[y] /= self.lut[y].norm()
            # print('lut', torch.sum(self.lut))
        return grad_inputs, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(inputs, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average



    def forward(self, inputs, targets, lut):
        inputs = oim(inputs, targets, lut, momentum=self.momentum)
        inputs *= self.scalar
        # print('inputs', torch.sum(inputs))
        loss = F.cross_entropy(inputs, targets.long(), weight=self.weight,
                               size_average=self.size_average)
        return loss, inputs



def oim_loss(inputs, targets, lut):
    oim_fn = OIMLoss(2048, 5532)
    loss = oim_fn(inputs, targets, lut)
    return loss[0]