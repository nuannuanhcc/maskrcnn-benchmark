# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from maskrcnn_benchmark.structures.image_list import resize_to_image_test


def compute_on_dataset(reid_model, model, data_loader, device, mode):
    reid_model.eval()
    model.eval()
    results_dict = []
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            if mode == 'query':
                results = model(images)
            if mode == 'test':
                results = model(images)
            images_reid, results = resize_to_image_test(images.tensors, results, targets, mode=mode)
            if images_reid is None:
                global_feature = None  #  if there no prediction image
            else:
                images_reid = [i.to(device) for i in images_reid]
                global_feature = reid_model(images_reid, results, i, mode=mode)
            outputs = results
            for j, res in enumerate(outputs):
                outputs[j].add_field('reid_feature', global_feature)  # correspondingly
                outputs = [o.to(cpu_device) for o in outputs]
            results_dict.append(outputs[0])
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        reid_model,
        model,
        data_loaders,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    modes = ['test', 'query']
    # modes = ['query', 'test']
    dataset_all=[]
    predictions_all = []
    for mode, data_loader in zip(modes, data_loaders):
        # for i, batch in enumerate(tqdm(data_loader)):
        #     images, targets, image_ids = batch
        print(mode, len(data_loader.dataset))
        device = torch.device(device)
        num_devices = get_world_size()
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(data_loader.dataset)))
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        dataset = data_loader.dataset
        predictions = compute_on_dataset(reid_model, model, data_loader, device, mode)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(data_loader.dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(data_loader.dataset),
                num_devices,
            )
        )

        # predictions = _accumulate_predictions_from_multiple_gpus(predictions) # we use 1 gpus

        if not is_main_process():
            return

        if output_folder:
            if mode == 'test':
                torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
            elif mode == 'query':
                torch.save(predictions, os.path.join(output_folder, "evaluations.pth"))
            else:
                raise KeyError(mode)
        dataset_all.append(dataset)
        predictions_all.append(predictions)

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset_all, predictions=predictions_all, output_folder=output_folder, **extra_args)
