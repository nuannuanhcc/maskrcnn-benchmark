# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.reid_backbone import build_reid_model
from maskrcnn_benchmark.utils.model_serialization import load_state_dict

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    import random
    import torch.backends.cudnn as cudnn
    import numpy as np
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed + 1)
    random.seed(seed + 2)
    np.random.seed(seed + 3)
    print('use seed')
    cudnn.deterministic = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # load a reid model
    reid_model = build_reid_model(cfg)
    reid_model.to(cfg.MODEL.DEVICE)
    print('#######loading from {}#######'.format(cfg.REID.TEST.WEIGHT))
    f = torch.load(cfg.REID.TEST.WEIGHT, map_location=torch.device("cpu"), )
    if 'model' in f:
        load_state_dict(reid_model, f['model'])
    else:
        reid_model.load_state_dict(f, strict=False)


    subdir, model_th = os.path.split(cfg.SUBDIR)
    output_dir = os.path.join(cfg.OUTPUT_DIR, subdir)
    print('Checkpoint dir: {}'.format(output_dir))

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)

    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        output_folder = os.path.join(output_dir, "inference", dataset_names[0].split('_')[0])
        mkdir(output_folder)


    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)

    if cfg.CIRCLE:
        for i in range(100):
            # pmodel = os.path.join(output_dir, model_th[:-11] + str(int(model_th[-11:-4]) + 2500 * i).zfill(7) + '.pth')
            pmodel = os.path.join(output_dir, model_th[:-11] + str(int(model_th[-11:-4]) + 1500 * i).zfill(7) + '.pth')
            _ = checkpointer.load(pmodel)
            inference(
                reid_model,
                model,
                data_loaders_val,
                dataset_name=dataset_names[0].split('_')[0],
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
                )
            synchronize()
    else:
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        inference(
            reid_model,
            model,
            data_loaders_val,
            dataset_name=dataset_names[0].split('_')[0],
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

if __name__ == "__main__":
    main()
