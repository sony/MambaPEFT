# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from copy import deepcopy

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.runner import Runner
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='whether to auto scale the learning rate according to the '
        'actual batch size and the original batch size.')
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='whether to disable the persistent_workers option in dataloaders.'
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get('type', 'OptimWrapper')
        assert optim_wrapper in ['OptimWrapper', 'AmpOptimWrapper'], \
            '`--amp` is not supported custom optimizer wrapper type ' \
            f'`{optim_wrapper}.'
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.setdefault('loss_scale', 'dynamic')

    # resume training
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # enable auto scale learning rate
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # set dataloader args
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        persistent_workers=True,
        collate_fn=dict(type='default_collate'),
    )
    if digit_version(TORCH_VERSION) < digit_version('1.8.0'):
        default_dataloader_cfg.persistent_workers = False

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        if isinstance(cfg.get(field, None), list): #added by yoshimura
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False
        if args.no_persistent_workers:
            cfg[field]['persistent_workers'] = False

    set_default_dataloader_cfg(cfg, 'train_dataloader')
    set_default_dataloader_cfg(cfg, 'val_dataloader')
    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Added by Yoshimura for Vtab-1K
    if cfg.get("sub_dataset_name", None) is not None:
        sub_dataset_name = cfg.get("sub_dataset_name", None)
        NUM_CLASSES = {
            'cifar':100,
            'caltech101':102,
            'dtd':47,
            'oxford_flowers102':102,
            'oxford_iiit_pet':37,
            'svhn':10,
            'sun397':397,
            'patch_camelyon':2,
            'eurosat':10,
            'resisc45':45,
            'diabetic_retinopathy':5,
            'clevr_count':8,
            'clevr_dist':6,
            'dmlab':6,
            'kitti':4,
            'dsprites_loc':16,
            'dsprites_ori':16,
            'smallnorb_azi':18,
            'smallnorb_ele':9,
        }
        num_class = NUM_CLASSES[sub_dataset_name]
        cfg.data_preprocessor.num_classes=num_class
        cfg.model.head.num_classes=num_class
        cfg.train_dataloader.dataset.sub_dataset_name=sub_dataset_name
        cfg.val_dataloader.dataset.sub_dataset_name=sub_dataset_name
        cfg.test_dataloader.dataset.sub_dataset_name=sub_dataset_name
        cfg.default_hooks.checkpoint.filename_tmpl='epoch_{}_'+sub_dataset_name+'.pth'

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
