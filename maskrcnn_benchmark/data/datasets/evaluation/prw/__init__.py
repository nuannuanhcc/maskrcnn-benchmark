

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#from .coco import COCODataset
#from .concat_dataset import ConcatDataset
#from .sysu import SYSUDataset ###
#from .voc import PascalVOCDataset ###

from .reid_dataset import PReIDDataset

import logging

from .prw_eval import do_prw_evaluation



def prw_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("sysu evaluation doesn't support box_only, ignored.")
    logger.info("performing sysu evaluation, ignored iou_types.")
    return do_prw_evaluation(
        datasets=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )


__all__ = ["PReIDDataset"]