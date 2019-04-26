# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn

from torch.autograd import Function
import torch.nn.functional as F

from IPython import embed

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


# def hard_example_mining(dist_mat, labels, return_inds=False):
#     """For each anchor, find the hardest positive and negative sample.
#     Args:
#       dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
#       labels: pytorch LongTensor, with shape [N]
#       return_inds: whether to return the indices. Save time if `False`(?)
#     Returns:
#       dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
#       dist_an: pytorch Variable, distance(anchor, negative); shape [N]
#       p_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
#       n_inds: pytorch LongTensor, with shape [N];
#         indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
#     NOTE: Only consider the case in which all labels have same num of samples,
#       thus we can cope with all anchors in parallel.
#     """
#
#     assert len(dist_mat.size()) == 2
#     assert dist_mat.size(0) == dist_mat.size(1)
#     N = dist_mat.size(0)
#
#     # shape [N, N]
#     is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
#     is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
#
#     # `dist_ap` means distance(anchor, positive)
#     # both `dist_ap` and `relative_p_inds` with shape [N, 1]
#     dist_ap, relative_p_inds = torch.max(
#         dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
#     # `dist_an` means distance(anchor, negative)
#     # both `dist_an` and `relative_n_inds` with shape [N, 1]
#     dist_an, relative_n_inds = torch.min(
#         dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
#     # shape [N]
#     dist_ap = dist_ap.squeeze(1)
#     dist_an = dist_an.squeeze(1)
#
#     if return_inds:
#         # shape [N, N]
#         ind = (labels.new().resize_as_(labels)
#                .copy_(torch.arange(0, N).long())
#                .unsqueeze(0).expand(N, N))
#         # shape [N, 1]
#         p_inds = torch.gather(
#             ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
#         n_inds = torch.gather(
#             ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
#         # shape [N]
#         p_inds = p_inds.squeeze(1)
#         n_inds = n_inds.squeeze(1)
#         return dist_ap, dist_an, p_inds, n_inds
#
#     return dist_ap, dist_an


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # print ('hard_example_mining ...')
    # embed() #

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    # dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap, relative_p_inds = torch.max((dist_mat * is_pos.type(torch.cuda.FloatTensor)).contiguous().view(N, -1), 1, keepdim=True)  #
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    # dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    dist_an, relative_n_inds = torch.min((dist_mat * is_neg.type(torch.cuda.FloatTensor) + 10 * is_pos.type(torch.cuda.FloatTensor)).contiguous().view(N, -1), 1, keepdim=True)  #
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    ####################################################
    index = (labels >= 0).nonzero().squeeze()  #
    dist_ap = dist_ap[index]
    dist_an = dist_an[index]
    relative_p_inds = relative_p_inds[index]
    relative_n_inds = relative_n_inds[index]
    ####################################################
    # print ('hard_example_mining ...')
    # embed() #

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

#################################################################################
class BF(Function):
    def __init__(self, lut, queue, num_gt, momentum):
        super(BF, self).__init__()
        self.lut = lut
        self.queue = queue
        self.momentum = momentum  # TODO: use exponentially weighted average
        self.num_gt = num_gt

    def forward(self, *inputs):
        inputs, targets = inputs
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.queue.t())
        for i, (x, y) in enumerate(zip(inputs, targets)):
            if y == -1:
                tmp = torch.cat((self.queue[1:], x.view(1, -1)), 0)
                self.queue[:, :] = tmp[:, :]
            elif y < len(self.lut):
                if i < self.num_gt:
                    self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
                    # self.lut[y] /= self.lut[y].norm()
            else:
                continue
        return torch.cat((outputs_labeled, outputs_unlabeled), 1)

    def backward(self, *grad_outputs):
        grad_outputs, = grad_outputs
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((self.lut, self.queue), 0))

        for i, (x, y) in enumerate(zip(inputs, targets)):
            if y == -1:
                tmp = torch.cat((self.queue[1:], x.view(1, -1)), 0)
                self.queue[:, :] = tmp[:, :]
            elif y < len(self.lut):
                if i < self.num_gt:
                    self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
                    # self.lut[y] /= self.lut[y].norm()
            else:
                continue

        return grad_inputs, None
#################################################################################

def triplet_loss(feature, pid_label, lut1, lut2, queue, momentum=0.0, iteration=0, margin=0.5):
    triplet_fn = TripletLoss(margin)  # triplet loss

    global_feat = []
    global_pid = []
    for feat, label in zip(feature, pid_label):
        # print ('triplet_loss 1 ...')
        # embed()
        global_feat.append(feat.unsqueeze(dim=0))
        global_feat.append(lut1[label].clone().unsqueeze(dim=0))
        global_feat.append(lut2[label].clone().unsqueeze(dim=0))
        global_pid.append(label.unsqueeze(dim=0))
        global_pid.append(label.unsqueeze(dim=0))
        global_pid.append(label.unsqueeze(dim=0))



    feat_triplet = torch.cat(global_feat)
    label_triplet = torch.cat(global_pid)

    num_gt = len(pid_label)
    if iteration % 2 == 1:
        reid_result = BF(lut1, queue, num_gt, momentum)(feature, pid_label)
    elif iteration % 2 == 0:
        reid_result = BF(lut2, queue, num_gt, momentum)(feature, pid_label)


    # for feat, label in zip(feature, pid_label):
    #     if flag:
    #         lut1[label] = feat
    #     else:
    #         lut2[label] = feat

    if len(global_pid) < 6:
        return torch.tensor(1.0).cuda()

    # print ('triplet_loss 2 ...')
    # embed()

    tripletloss = triplet_fn(feat_triplet, label_triplet)[0]



    # print ('triplet_loss 3 ...')
    # embed()


    return tripletloss