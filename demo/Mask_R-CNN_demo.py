#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

pylab.rcParams['figure.figsize'] = 20, 12

def load(url):
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


#  raw image
pil_image = Image.open('/unsullied/sharefs/_research_video/VideoData/users/yejiacheng/pytorch/dataset/person_search/coco_format_sysu/test/s3289.jpg').convert("RGB")
pil_image.save('a.jpg')

#  mask image
config_file = "../configs/demo_mask_R_50_FPN.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
coco_demo = COCODemo(cfg,min_image_size=800,confidence_threshold=0.7,)
# image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
image = np.array(pil_image)[:, :, [2, 1, 0]]
# imshow(image)
predictions = coco_demo.run_on_opencv_image(image)
# imshow(predictions)
im=Image.fromarray(predictions[:, :, [2, 1, 0]])
im.save('b.jpg')

#  detetction image
config_file = "../configs/demo_faster_R_50_FPN.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
coco_demo = COCODemo(cfg,min_image_size=800,confidence_threshold=0.7,)
predictions = coco_demo.run_on_opencv_image(image)
# imshow(predictions)
im=Image.fromarray(predictions[:, :, [2, 1, 0]])
im.save('c.jpg')

#  keypoint image
config_file = "../configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
cfg.merge_from_list(["MODEL.MASK_ON", False])

coco_demo = COCODemo(cfg,min_image_size=800,confidence_threshold=0.7,)

# image = load("http://farm9.staticflickr.com/8419/8710147224_ff637cc4fc_z.jpg")
predictions = coco_demo.run_on_opencv_image(image)
# imshow(predictions)
im=Image.fromarray(predictions[:, :, [2, 1, 0]])
im.save('d.jpg')



