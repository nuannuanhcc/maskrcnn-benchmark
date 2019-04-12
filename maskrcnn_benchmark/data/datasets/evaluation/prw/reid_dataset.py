import os

import cv2
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import numpy as np
from IPython import embed

import json
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class PReIDDataset(torch.utils.data.Dataset):

    CLASSES = ("__background__ ", 'person') ###

    def __init__(self, root, ann_file, split, mode='test', transforms=None):

        self.mode = mode
        self.gallery_size = 5000
        assert self.mode != 'test', "{} mode error!".format(self.mode)
        self.root = root
        self.anno_dir = '/home/yejiacheng/tmp/maskrcnn_master/datasets/sysu/annotations'
        self.anno_file = ann_file
        self.anno = os.path.join(self.anno_dir, self.anno_file)
        self.split = split
        self.transforms = transforms
        self.train_DF = '/unsullied/sharefs/_research_video/VideoData/users/yejiacheng/pytorch/dataset/person_search/PRW/SIPN_annotation/trainAllDF.csv'
        self.test_DF = '/unsullied/sharefs/_research_video/VideoData/users/yejiacheng/pytorch/dataset/person_search/PRW/SIPN_annotation/testAllDF.csv'
        self.query_DF = '/unsullied/sharefs/_research_video/VideoData/users/yejiacheng/pytorch/dataset/person_search/PRW/SIPN_annotation/queryDF.csv'
        self.gallery = '/unsullied/sharefs/_research_video/VideoData/users/yejiacheng/pytorch/dataset/person_search/PRW/SIPN_annotation/q_to_g{}DF.csv'.format(self.gallery_size)

        self.demo = False ###############

        if self.demo:
            self.pid = 'pid_0.csv'
        #self.annotations = os.path.join(self.root, 'annotations')
        #self.imgpath = os.path.join(self.root, 'images')
        # sort indices for reproducible results
        #self.ids = sorted(self.ids)  # The id list of dataset
            self.pid_file = os.path.join(self.anno_dir, 'pids', self.pid)
            query_box = pd.read_csv(self.pid_file)
            imname = query_box['imname']
            self.ids = np.array(imname.squeeze()).tolist()
        else:
            #"""
            with open(self.anno) as json_anno:
                anno_dict = json.load(json_anno)
            self.ids = [img['file_name'] for img in anno_dict['images']]
            #"""

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = PReIDDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

        if self.split == 'train' or self.split == 'val':
            self.all_boxes = pd.read_csv(self.train_DF)
        elif self.split == 'test':
            if self.mode == 'gallery':
                pass
            elif self.mode == 'query':
                #self.all_boxes = query_box
                self.all_boxes = pd.read_csv(self.query_DF)
                #self.all_boxes = pd.read_csv(self.test_DF)
            else:
                raise (KeyError(self.mode))
        else:
            raise(KeyError(self.split))

    # as you would do normally

    def __getitem__(self, index):
        # load the image as a PIL Image
        img_id = self.ids[index]
        im_path = os.path.join(self.root, img_id)
        img = Image.open(im_path).convert("RGB")
        #img = cv2.imread(im_path).astype(np.float32)
        #orig_shape = img.size
        #print (orig_shape)
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.

        #print (type(boxes))
        #print (boxes)
        #boxes = [obj["bbox"] for obj in anno]
        #boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        #target = BoxList(boxes, img.size[:2], mode="xywh").convert("xyxy")
        #embed()
        #target = BoxList(boxes, orig_shape, mode="xyxy")
        # and labels
        #classes = [obj["category_id"] for obj in anno]
        # classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        #classes = boxes_df.loc[:, 'cls_id'].copy()
        #classes = classes.values.tolist()
        #print (type(classes))
        #print (classes)
        #classes = torch.Tensor(classes)
        #target.add_field("labels", classes)


        #pid = boxes_df.loc[:, 'pid'].copy()
        #pid = pid.values.tolist()
        #print (type(pid))
        #print (pid)
        #pid = torch.Tensor(pid)
        #target.add_field("pid", pid)

        # create a BoxList from the boxes
        #boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        #boxlist.add_field("labels", labels)

        #masks = [obj["segmentation"] for obj in anno]
        #masks = SegmentationMask(masks, img.size)
        #target.add_field("masks", masks)

        target = self.get_groundtruth(index)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # return the image, the boxlist and the idx in your dataset
        return img, target, index

    def get_groundtruth(self, index):
        # img_id = self.ids[index]
        boxes_df = self.all_boxes.query('index==@index')
        # boxes_df = self.all_boxes.query('imname==@img_id and pid==@index') ###
        #boxes_df = self.all_boxes.query('pid==@index')
        #boxes_df = self.all_boxes.iloc[index] ###
        #print (boxes_df)
        #embed() ###
        boxes = boxes_df.loc[:, 'x1': 'pid'].copy()
        #boxes = boxes_df.copy()
        boxes.loc[:, 'del_x'] += boxes.loc[:, 'x1']
        boxes.loc[:, 'del_y'] += boxes.loc[:, 'y1']
        boxes = boxes.values.astype(np.float32)
        boxes = boxes.tolist()
        anno = self._preprocess_annotation(boxes)

        orig_shape = self.get_img_info(index)

        (width, height) = orig_shape['width'], orig_shape['height']
        #print ('anno[\'pid\']: ', anno['pid'])
        #print ('anno[\'boxes\']: ', anno['boxes'])
        #embed() ###
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        #target.add_field("img", anno["img_name"])
        target.add_field("labels", anno["labels"])
        target.add_field("pid", anno["pid"])
        target.add_field("difficult", anno["difficult"])
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        pid = []
        #img_name = []
        name = 'person'
        difficult = False
        difficult_boxes = []
        for obj in target:
            #print (obj)
            #difficult = int(obj.find("difficult").text) == 1

            #name = obj.find("name").text.lower()
            #bb = obj.find("bndbox")
            bndbox = tuple(
                map(
                    int,
                    [
                        obj[0],
                        obj[1],
                        obj[2],
                        obj[3],
                    ],
                )
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            pid.append(int(obj[-1]))
            #img_name.append(str(obj[0])) ###
            difficult_boxes.append(difficult)

        #size = target.find("size")
        #im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            #"img_name": torch.tensor(img_name, dtype=torch._tensor_str),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "pid": torch.tensor(pid, dtype=torch.int32),
            "difficult": torch.tensor(difficult_boxes),
            #"im_info": im_info,
        }
        return res

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_id = self.ids[index]
        im_path = os.path.join(self.root, img_id)
        img = cv2.imread(im_path).astype(np.float32)
        orig_shape = img.shape

        return {"height": orig_shape[0], "width": orig_shape[1]}

    def map_class_id_to_class_name(self, class_id):
        return PReIDDataset.CLASSES[class_id]

