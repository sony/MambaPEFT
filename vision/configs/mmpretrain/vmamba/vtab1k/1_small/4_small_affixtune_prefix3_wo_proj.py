_base_ = [
    '../../../_my_base_/datasets/vtab1k_bs32_224.py',
    '../../../_my_base_/schedules/vtab1k_bs32_adamw.py',
    '../../../_my_base_/default_runtime.py'
]

sub_dataset_name = 'cifar'
data_root = 'data/vtab-1k/'

checkpoint = "work_dirs/weights/backbone/vmamba/vssm_small_0229_ckpt_epoch_222.pth" 

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VSSMPEFTModel',
        num_virtual_tokens=3,
        prefix_projection=False,
        prefix_tuning=True,
        prefix_type='inner_dual_prefix',
        module=dict(
            type='Backbone_VSSM',
            depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3, 
            patch_size=4, in_chans=3, num_classes=1000, 
            ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
            ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, 
            ssm_init="v0", forward_type="v05_noz", 
            mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
            patch_norm=True, norm_layer="ln2d", 
            downsample_version="v3", patchembed_version="v2", 
            use_checkpoint=False, posembed=False, imgsize=224, 
            pretrained=checkpoint,
        ),
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=100,
        in_channels=96*8,
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
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=1e-4,
        eps=1e-8,),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=1.0,
        custom_keys={
            '.A_b_log':
            dict(decay_mult=0.0),
            '.A_log':
            dict(decay_mult=0.0),
            '.D.':
            dict(decay_mult=0.0),
            '.D_b.':
            dict(decay_mult=0.0),
            '.absolute_pos_embed':
            dict(decay_mult=0.0),
            '.cls_token':
            dict(decay_mult=0.0),
            '.dist_token':
            dict(decay_mult=0.0),
            '.pos_embed':
            dict(decay_mult=0.0),
            '.relative_position_bias_table':
            dict(decay_mult=0.0),
            'A_b_log_scan_addi':
            dict(decay_mult=0.0, lr_mult=1.0),
            'A_log_scan_addi':
            dict(decay_mult=0.0, lr_mult=1.0),
            'learnable_A':
            dict(decay_mult=10.0, lr_mult=1.0),
            'learnable_D':
            dict(decay_mult=10.0, lr_mult=1.0),
            'learnable_cls_token':
            dict(decay_mult=10.0, lr_mult=1.0),
            'learnable_conv1d':
            dict(decay_mult=10.0, lr_mult=1.0),
            'learnable_patch_embed':
            dict(decay_mult=10.0, lr_mult=1.0),
            'lora_B':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_C':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_X':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_Z':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_d':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_dt':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_in_proj':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_out_proj':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_patch_embed':
            dict(decay_mult=1.0, lr_mult=1.0),
            'lora_x_proj':
            dict(decay_mult=1.0, lr_mult=0.1),
            'prefix_encoder':
            dict(decay_mult=1.0, lr_mult=1.0),
            'prompt_encoder':
            dict(decay_mult=1.0, lr_mult=1.0),
            'x_proj_b_scan_addi':
            dict(decay_mult=1.0, lr_mult=1.0),
            'x_proj_scan_addi':
            dict(decay_mult=1.0, lr_mult=1.0)
        }),
    #clip_grad=dict(max_norm=5.0),
)

# runtime settings
# custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1000, val_begin=1000) # No validation to save time

default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHookWithInvalidLossChecker',
        interval=10,
        save_optimizer=True,
        max_keep_ckpts=1,
        filename_tmpl='epoch_{}_'+sub_dataset_name+'.pth',
    ),
)
