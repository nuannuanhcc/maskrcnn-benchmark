# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import division
from maskrcnn_benchmark.config import cfg
import torch.nn.functional as F
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
import torch


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def to_image_list(tensors, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))  # 3 * w * h
        # TODO Ideally, just remove this and let me model handle arbitrary
        # input size
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size  # 4 * 3 * w * h
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]

        return ImageList(batched_imgs, image_sizes)
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def iou_max(boxesA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    ious = []
    for i in boxesA:
        ious.append(iou(i, boxB))
    index_max = ious.index(max(ious))
    # print(max(ious))
    return index_max

def stn(x, box):
    x = x.squeeze()
    if box.dim() == 1:
        box = box.expand(1, -1)
    if x.dim() == 3:  # image
        x = x.expand(box.shape[0], -1, -1, -1)
    if x.dim() == 2:  # mask
        x = x.expand(box.shape[0], 1, -1, -1)
    temp_box = box.float().clone()
    # print ('stn embedding ...')
    # normalization the weight and height 0~w/h => -1~1
    # min~max => a~b  x_norm = (b-a)/(max-min)*(x-min)+a
    temp_box[:, (0, 2)] = 2 * (temp_box[:, (0, 2)]) / x.shape[-1] - 1
    temp_box[:, (1, 3)] = 2 * (temp_box[:, (1, 3)]) / x.shape[-2] - 1
    # calculate the affine parameter
    theta = torch.zeros((x.shape[0], 6)).cuda()
    theta[:, 0] = (temp_box[:, 2] - temp_box[:, 0]) / 2
    theta[:, 2] = (temp_box[:, 2] + temp_box[:, 0]) / 2
    theta[:, 4] = (temp_box[:, 3] - temp_box[:, 1]) / 2
    theta[:, 5] = (temp_box[:, 3] + temp_box[:, 1]) / 2
    theta = theta.view(-1, 2, 3)
    # new_size is changable
    new_size = torch.Size([*x.shape[:2], 384, 128])
    grid = F.affine_grid(theta, new_size)
    x = F.grid_sample(x, grid)
    return x

def transform_from_detect_to_reid(images):
    mean_d = torch.tensor(cfg.INPUT.PIXEL_MEAN, dtype=torch.float32)
    std_d = torch.tensor(cfg.INPUT.PIXEL_STD, dtype=torch.float32)
    mean_r = torch.tensor(cfg.REID.INPUT.PIXEL_MEAN, dtype=torch.float32)
    std_r = torch.tensor(cfg.REID.INPUT.PIXEL_STD, dtype=torch.float32)

    img = images.cpu()
    img = img.mul_(std_d[:, None, None]).add_(mean_d[:, None, None]) / 255.0
    img = img[:, [2, 1, 0], :, :].cpu()
    img = img.sub_(mean_r[:, None, None]).div_(std_r[:, None, None])
    device = cfg.MODEL.DEVICE
    img = img.to(device)
    # img = transforms.ToPILImage()(img[0]).convert('RGB')
    # img.save('a.jpg')
    return img

def mask_compute(result, image, images_reid):
    # pad the mask with the image's size
    masks = result.get_field("mask")
    masks_pad_shape = (masks.shape[0], masks.shape[1],)+(image.shape[-2], image.shape[-1])
    masks_pad_img = image[0].new(*masks_pad_shape).zero_()
    # masks_pad_img = torch.zeros((masks_pad_shape), dtype=torch.uint8).cuda()
    for img, img_pad in zip(masks, masks_pad_img):
        img_pad[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # results[0].add_field("mask", masks_pad_img)
    masks_reid = []
    for mask, bbox in zip(masks_pad_img, result.bbox):
        masks_reid.append(stn(mask, bbox))
    masks_reid = torch.cat(masks_reid, dim=0)
    images_maks_reid = torch.mul(masks_reid, images_reid)
    return images_maks_reid

def resize_to_image(images, targets, results):
    images = transform_from_detect_to_reid(images)
    reid_images = []
    reid_labels = []
    reid_masks = []
    outputs = []
    for image, target, result in zip(images, targets, results):
        result_list = []
        for i in range(target.bbox.shape[0]):
            if target.extra_fields['pid'][i] < 0:
                continue
            elif target.extra_fields['pid'][i] == 5532:
                continue
            if cfg.MODEL.MASK_ON:
                # filter other classes
                label = result.get_field('labels')
                keep = torch.nonzero(label == 1).squeeze(1)
                result = result[keep]
            if len(result.bbox) == 0:
                print('###### there is no predict_box #####', target.extra_fields['pid'])
                continue
            idx = iou_max(result.bbox, target.bbox[i])
            result_idx = result.get_items(range(idx, idx + 1))
            result_idx.extra_fields['pid'] = torch.tensor([target.extra_fields['pid'][i]])
            result_list.append(result_idx)  # pid,labels,scores
        if result_list == []:
            continue
        result_list = cat_boxlist(result_list)
        reid_image = stn(image, result_list.bbox)
        reid_images.append(reid_image)
        reid_labels.append(result_list.extra_fields['pid'])
        if cfg.MODEL.MASK_ON:
            images_mask_reid = mask_compute(result_list, image, reid_image)
            reid_masks.append(images_mask_reid)
    if reid_images == []:
        print('######### wrong ################')
        return None, None
    outputs.append(torch.cat(reid_images))
    if cfg.MODEL.MASK_ON:
        outputs.append(torch.cat(reid_masks))
    return outputs, torch.cat(reid_labels)


def resize_to_image_test(images, results, target, mode):
    images = transform_from_detect_to_reid(images)
    output=[]
    if not cfg.MODEL.MASK_ON:
        if mode == 'query':
            proposal = target[0].bbox
            results = target
        if mode == 'test':
            proposal = results[0].bbox
            if len(proposal) == 0:
                return None, results
        images_reid = stn(images, proposal)
        output.append(images_reid)

    if cfg.MODEL.MASK_ON:
        if mode == 'query':
            proposal = target[0].bbox
            images_reid = stn(images, proposal)
            output.append(images_reid)
            return output, target
        if mode == 'test':
            # filter other classes
            label = results[0].get_field('labels')
            keep = torch.nonzero(label == 1).squeeze(1)
            results[0] = results[0][keep]

            proposal = results[0].bbox
            if len(proposal) == 0:
                return None, results

            images_reid = stn(images, proposal)
            images_mask_reid = mask_compute(results[0], images, images_reid)
            del results[0].extra_fields['mask']
            output.append(images_mask_reid)
            output.append(images_reid)
        return output, results

    #     if mode == 'query':
    #         mask = results[0].get_field('mask')
    #         target.add_field('mask', mask)
    #         proposal = target[0].bbox





