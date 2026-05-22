import copy
import logging
from collections import defaultdict, OrderedDict
from mmengine.logging import MMLogger, print_log
from typing import Dict, List, Optional, Sequence, Union
from mmdet.evaluation.metrics.coco_metric import CocoMetric
from mmdet.registry import METRICS
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)
from typing import Any, List, Optional, Sequence, Union
from mmdet.structures.mask import encode_mask_results
from torch import Tensor
from mmengine.structures import BaseDataElement
import torch
from mmengine.fileio import dump, get_local_path, load
from mmdet.datasets.api_wrappers import COCO

def _to_cpu(data: Any) -> Any:
    """Transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data



@METRICS.register_module()
class DGCocoMetric(CocoMetric):
    def __init__(
        self,
        ann_files: dict = None,
        dataset_keys=[],
        mean_used_keys=[],
        **kwargs
    ):
        """
        Args:
            ann_files (dict): key=dataset_key, value=path to ann_file
            dataset_keys (list): used to determine dataset_key from img_path
            mean_used_keys (list): datasets used to compute mean_*
        """
        super().__init__(**kwargs)
        self.ann_files = ann_files or {}
        self.dataset_keys = dataset_keys
        self.mean_used_keys = mean_used_keys or dataset_keys

        # Save predictions by dataset key
        self.grouped_results = defaultdict(list)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Group prediction results by dataset key based on img_path."""
      
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()

            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            gt['anns'] = []
            boxes = data_sample['gt_instances']['bboxes'].detach().cpu().numpy()
            labels = data_sample['gt_instances']['labels'].detach().cpu().numpy()
            for bbox, label in zip(boxes, labels):
                gt['anns'].append({'bbox': bbox, 'bbox_label': label})

            # determine dataset key based on img_path
            img_path = data_sample.get("img_path", "")
            dataset_key = "unknown"
            for key in self.dataset_keys:
                if key in img_path:
                    dataset_key = key
                    break
            self.grouped_results[dataset_key].append((gt, result))
    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.

        Returns:
            dict: Evaluation metrics dict.
        """
        logger = MMLogger.get_current_instance()

        if len(self.grouped_results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `grouped_results`. Please '
                'ensure that the `process` method was called correctly.',
                logger=logger,
                level=logging.WARNING)

        # -------- 1. Flatten grouped_results to list for collection -------- #
        flat_results = []
        for key, samples in _to_cpu(self.grouped_results).items():
            for sample in samples:
                flat_results.append((key, sample))

        # -------- 2. Collect results across all ranks -------- #
        all_parts = collect_results(
            flat_results, size=size, device=self.collect_device)

        # -------- 3. On main process, regroup results by dataset key -------- #
        if is_main_process():
            merged_grouped_results = defaultdict(list)
            for key, sample in all_parts:
                merged_grouped_results[key].append(sample)
            self.grouped_results = merged_grouped_results

            # -------- 4. Compute evaluation metrics -------- #
            _metrics = self.compute_group_metrics(self.grouped_results)

            if self.prefix:
                _metrics = {
                    f"{self.prefix}/{k}": v for k, v in _metrics.items()
                }

            metrics = [_metrics]
        else:
            metrics = [None]

        # -------- 5. Broadcast final result to all processes -------- #
        broadcast_object_list(metrics)

        # -------- 6. Clear temporary state -------- #
        self.grouped_results.clear()

        return metrics[0]



    def compute_group_metrics(self, results: list) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        metrics = defaultdict(list)
        #metrics_type2mean = defaultdict(list)

        for key, samples in self.grouped_results.items():
            print_log(f"\n--------- Evaluating {key} ---------", logger)

            ann_file = self.ann_files.get(key, None)
            if ann_file is None:
                print_log(f"[Warning] ann_file for dataset_key '{key}' not provided, skipping.", logger)
                continue
            coco_metric = CocoMetric(
                ann_file=ann_file,
                metric=self.metrics,
                classwise=self.classwise,
                format_only=False,
                iou_thrs=self.iou_thrs,
                proposal_nums=self.proposal_nums,
                outfile_prefix=None,
                backend_args=self.backend_args,
            )
            coco_metric.dataset_meta = self.dataset_meta
            coco_metric.results = samples

            key_metrics = coco_metric.compute_metrics(samples)

            for k, v in key_metrics.items():
                metrics[f"{key}_{k}"] = v


        return metrics
