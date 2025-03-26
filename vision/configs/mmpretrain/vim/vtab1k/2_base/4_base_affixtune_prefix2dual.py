_base_ = [
    '../../../_my_base_/datasets/vtab1k_bs32_224.py',
    '../../../_my_base_/schedules/vtab1k_bs32_adamw.py',
    '../../../_my_base_/default_runtime.py'
]

sub_dataset_name = 'cifar'
data_root = 'data/vtab-1k/'

checkpoint = "work_dirs/weights/backbone/vim/vim_b_midclstok_81p9acc.pth" 


# model settings
model = dict(
    type='ImageClassifier',
    # init_cfg = dict(type="RevisePretrained", checkpoint=checkpoint2,
    #         ),
    backbone=dict(
        type='VimPEFTModel',
        prefix_tuning=True, prefix_type="inner_dual_prefix", num_virtual_tokens=2, 
        module=dict(
            type='VisionMamba',
            patch_size=16, embed_dim=768, depth=24, rms_norm=True, residual_in_fp32=True, 
            fused_add_norm=False, final_pool_type='mean', if_abs_pos_embed=True, if_rope=False, 
            if_rope_residual=False, bimamba_type="v2", if_cls_token=True, if_devide_out=True, use_middle_cls_token=True,
            drop_path_rate=0.1,
            init_cfg = dict(type="RevisePretrained", checkpoint=checkpoint, prefix=None, 
                revise_keys=[(r'^module\.', '')],
                # adding_prefix="backbone.",
                state_dict_key="model"
            ),
        ),
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=100,
        in_channels=768,
        # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ),
)

# dataset settings


_base_.train_dataloader.batch_size = 32
_base_.train_dataloader.dataset.data_root = data_root
_base_.val_dataloader.dataset.data_root = data_root
_base_.test_dataloader.dataset.data_root = data_root
_base_.train_dataloader.num_workers = 1


# schedule settings
optim_wrapper = dict(
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.A_b_log': dict(decay_mult=0.0),
            '.A_log': dict(decay_mult=0.0),
            '.D.': dict(decay_mult=0.0),
            '.D_b.': dict(decay_mult=0.0),
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.cls_token': dict(decay_mult=0.0),
            '.dist_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0),
        }),
    clip_grad=dict(max_norm=5.0),
)

# runtime settings
# custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1000, val_begin=1000)

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=10,
        save_optimizer=True,
        max_keep_ckpts=1,
        filename_tmpl='epoch_{}_'+sub_dataset_name+'.pth',
    ),
)