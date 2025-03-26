# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from mmengine.logging import print_log
from mmengine.registry import WEIGHT_INITIALIZERS, build_from_cfg
import re
from collections import OrderedDict
from mmengine.model.weight_init import update_init_info
from mmengine.runner.checkpoint import _load_checkpoint, load_state_dict

def _load_checkpoint_with_prefix(prefix, filename, map_location=None, state_dict_key="state_dict"):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location=map_location)

    if state_dict_key in checkpoint:
        state_dict = checkpoint[state_dict_key]
    else:
        state_dict = checkpoint

    if prefix is not None:
        if not prefix.endswith('.'):
            prefix += '.'
        prefix_len = len(prefix)
        
        state_dict = {
            k[prefix_len:]: v
            for k, v in state_dict.items() if k.startswith(prefix)
        }

        assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict
    
@WEIGHT_INITIALIZERS.register_module(name='RevisePretrained')
class RevisePretrainedInit:
    """Initialize module by loading a pretrained model.

    Args:
        checkpoint (str): the checkpoint file of the pretrained model should
            be load.
        prefix (str, optional): the prefix of a sub-module in the pretrained
            model. it is for loading a part of the pretrained model to
            initialize. For example, if we would like to only load the
            backbone of a detector model, we can set ``prefix='backbone.'``.
            Defaults to None.
        map_location (str): map tensors into proper locations. Defaults to cpu.
    """

    def __init__(self, checkpoint, prefix=None, adding_prefix=None, map_location='cpu', revise_keys=[(r'^module\.', '')], state_dict_key="state_dict"):
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location
        self.revise_keys = revise_keys
        self.state_dict_key = state_dict_key
        self.adding_prefix = adding_prefix

    def __call__(self, module):
        # from mmengine.runner.checkpoint import (_load_checkpoint_with_prefix,
        #                                         load_checkpoint,
        #                                         load_state_dict)
        # if self.prefix is None:
        #     print_log(f'load model from: {self.checkpoint}', logger='current')
        #     load_checkpoint(
        #         module,
        #         self.checkpoint,
        #         map_location=self.map_location,
        #         strict=False,
        #         revise_keys=self.revise_keys,
        #         logger='current')
        # else:
        print_log(
            f'load {self.prefix} in model from: {self.checkpoint}',
            logger='current')
        state_dict = _load_checkpoint_with_prefix(
            self.prefix, self.checkpoint, map_location=self.map_location, state_dict_key=self.state_dict_key)
        
        # strip prefix of state_dict
        metadata = getattr(state_dict, '_metadata', OrderedDict())

        if self.adding_prefix is not None:
            state_dict = OrderedDict(
                {self.adding_prefix+k: v
                for k, v in state_dict.items()})

        
        for p, r in self.revise_keys:
            state_dict = OrderedDict(
                {re.sub(p, r, k): v
                for k, v in state_dict.items()})
        # Keep metadata in state_dict
        state_dict._metadata = metadata

        load_state_dict(module, state_dict, strict=False, logger='current')

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: load from {self.checkpoint}'
        return info
