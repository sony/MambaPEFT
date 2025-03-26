# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import logging
import os.path as osp
import pickle
from math import inf
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union
import torch.distributed as dist
from mmengine.dist import is_main_process
from mmengine.fileio import FileClient, get_file_backend
from mmengine.logging import print_log
from mmengine.registry import HOOKS
from mmengine.utils import is_list_of, is_seq_of
from mmengine.hooks import Hook, CheckpointHook
import torch

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class CheckpointHookWithInvalidLossChecker(CheckpointHook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Defaults to -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Defaults to True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Defaults to True.
        save_param_scheduler (bool): Whether to save param_scheduler state_dict
            in the checkpoint. It is usually used for resuming experiments.
            Defaults to True.
        out_dir (str, Path, Optional): The root directory to save checkpoints.
            If not specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``. For example,
            if the input ``our_dir`` is ``./tmp`` and ``runner.work_dir`` is
            ``./work_dir/cur_exp``, then the ckpt will be saved in
            ``./tmp/cur_exp``. Defaults to None.
        max_keep_ckpts (int): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Defaults to -1, which means unlimited.
        save_last (bool): Whether to force the last checkpoint to be
            saved regardless of interval. Defaults to True.
        save_best (str, List[str], optional): If a metric is specified, it
            would measure the best checkpoint during evaluation. If a list of
            metrics is passed, it would measure a group of best checkpoints
            corresponding to the passed metrics. The information about best
            checkpoint(s) would be saved in ``runner.message_hub`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resuming checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Defaults to None.
        rule (str, List[str], optional): Comparison rule for best score. If
            set to None, it will infer a reasonable rule. Keys such as 'acc',
            'top' .etc will be inferred by 'greater' rule. Keys contain 'loss'
            will be inferred by 'less' rule. If ``save_best`` is a list of
            metrics and ``rule`` is a str, all metrics in ``save_best`` will
            share the comparison rule. If ``save_best`` and ``rule`` are both
            lists, their length must be the same, and metrics in ``save_best``
            will use the corresponding comparison rule in ``rule``. Options
            are 'greater', 'less', None and list which contains 'greater' and
            'less'. Defaults to None.
        greater_keys (List[str], optional): Metric keys that will be
            inferred by 'greater' comparison rule. If ``None``,
            _default_greater_keys will be used. Defaults to None.
        less_keys (List[str], optional): Metric keys that will be
            inferred by 'less' comparison rule. If ``None``, _default_less_keys
            will be used. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        filename_tmpl (str, optional): String template to indicate checkpoint
            name. If specified, must contain one and only one "{}", which will
            be replaced with ``epoch + 1`` if ``by_epoch=True`` else
            ``iteration + 1``.
            Defaults to None, which means "epoch_{}.pth" or "iter_{}.pth"
            accordingly.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.
        published_keys (str, List[str], optional): If ``save_last`` is ``True``
            or ``save_best`` is not ``None``, it will automatically
            publish model with keys in the list after training.
            Defaults to None.
            `New in version 0.7.1.`
    Examples:
        >>> # Save best based on single metric
        >>> CheckpointHook(interval=2, by_epoch=True, save_best='acc',
        >>>                rule='less')
        >>> # Save best based on multi metrics with the same comparison rule
        >>> CheckpointHook(interval=2, by_epoch=True,
        >>>                save_best=['acc', 'mIoU'], rule='greater')
        >>> # Save best based on multi metrics with different comparison rule
        >>> CheckpointHook(interval=2, by_epoch=True,
        >>>                save_best=['FID', 'IS'], rule=['less', 'greater'])
        >>> # Save best based on single metric and publish model after training
        >>> CheckpointHook(interval=2, by_epoch=True, save_best='acc',
        >>>                rule='less', published_keys=['meta', 'state_dict'])
    """
    

    def __init__(self,
                 loss_check_interval=10,
                 **kwargs) -> None:
        self.loss_check_interval = loss_check_interval
        super().__init__(**kwargs)

   
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs=Optional[dict]) -> None:
        """Save the checkpoint and synchronize buffers after each iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """

        if self.every_n_train_iters(runner, self.loss_check_interval):
            if not torch.isfinite(outputs['loss']):
                runner.logger.info(
                    f'Saving checkpoint at {runner.iter + 1} iterations')
                self._save_checkpoint(runner)
                if dist.is_initialized():
                    dist.barrier()
                assert torch.isfinite(outputs['loss']), \
                    runner.logger.info('loss become infinite or NaN!')

        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_train_iters(runner, self.interval) or \
                (self.save_last and
                 self.is_last_train_iter(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            self._save_checkpoint(runner)
