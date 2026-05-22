import os.path as osp
import copy
from typing import List, Optional, Sequence, Union

import numpy as np
from mmengine.logging import MMLogger, print_log

from mmdet.registry import METRICS
from mmdet.evaluation.metrics.voc_metric import VOCMetric
from collections import defaultdict
from .voc_metric import *

@METRICS.register_module()
class DGVOCMetric(VOCMetric):
    def __init__(self, dataset_keys=[], mean_used_keys=[], **kwargs):
        super().__init__(**kwargs)
        self.dataset_keys = dataset_keys
        if mean_used_keys:
            self.mean_used_keys = mean_used_keys
        else:
            self.mean_used_keys = dataset_keys

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            # TODO: Need to refactor to support LoadAnnotations
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            ann = dict(
                labels=gt_instances['labels'].cpu().numpy(),
                bboxes=gt_instances['bboxes'].cpu().numpy(),
                bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                labels_ignore=gt_ignore_instances['labels'].cpu().numpy())

            pred = data_sample['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            dataset_key = "unknown"
            for key in self.dataset_keys:
                if key in data_sample["img_path"]:
                    dataset_key = key
                    break

            self.results.append([dataset_key, ann, dets])
    def compute_metrics_(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)
        eval_results = OrderedDict()
        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_type = self.dataset_meta.get('dataset_type')
            if dataset_type in ['VOC2007', 'VOC2012']:
                dataset_name = 'voc'
                if dataset_type == 'VOC2007' and self.eval_mode != '11points':
                    warnings.warn('Pascal VOC2007 uses `11points` as default '
                                  'evaluate mode, but you are using '
                                  f'{self.eval_mode}.')
                elif dataset_type == 'VOC2012' and self.eval_mode != 'area':
                    warnings.warn('Pascal VOC2012 uses `area` as default '
                                  'evaluate mode, but you are using '
                                  f'{self.eval_mode}.')
            else:
                dataset_name = self.dataset_meta['classes']

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, _ = eval_map(
                    preds,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    dataset=dataset_name,
                    logger=logger,
                    eval_mode=self.eval_mode,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif self.metric == 'recall':
            gt_bboxes = [gt['bboxes'] for gt in gts]
            pr_bboxes = [pred[0] for pred in preds]
            recalls = eval_recalls(
                gt_bboxes,
                pr_bboxes,
                self.proposal_nums,
                self.iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(self.proposal_nums):
                for j, iou_thr in enumerate(self.iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results


    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        dataset_results = defaultdict(list)
        metrics = {}
        for result in results:
            dataset_results[result[0]].append(result[1:])
        metrics_type2mean = defaultdict(list)
        for key, key_result in dataset_results.items():
            logger: MMLogger = MMLogger.get_current_instance()
            print_log(f"----------metrics for {key}------------", logger)
            key_metrics = self.compute_metrics_(key_result)
            print_log(f"number of samples for {key}: {len(key_result)}")
            for k, v in key_metrics.items():
                metrics[f"{key}_{k}"] = v
                if key in self.mean_used_keys:
                    metrics_type2mean[k].append(v)
        for k, v in metrics_type2mean.items():
            metrics[f"mean_{k}"] = sum(v) / len(v)
        return metrics
