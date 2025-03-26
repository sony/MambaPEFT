_base_ = [
    '../../../_my_base_/datasets/vtab1k_bs32_224.py',
    '../../../_my_base_/schedules/vtab1k_bs32_adamw.py',
    '../../../_my_base_/default_runtime.py'
]

sub_dataset_name = 'cifar'
data_root = 'data/vtab-1k/'

checkpoint = "work_dirs/weights/backbone/vim/vim_s_midclstok_80p5acc.pth" 

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VimPEFTModel',
        additional_scan=False,
        bit=32,
        bit_B=32,
        bit_C=32,
        bit_X=32,
        bit_Z=32,
        bit_d=32,
        bit_dt=32,
        bit_in_proj=32,
        bit_patch_embed=32,
        bit_x_proj=32,
        dim=8,
        dim_B=4,
        dim_B_f=4,
        dim_C=4,
        dim_C_f=4,
        dim_X=8,
        dim_Z=8,
        dim_d=4,
        dim_d_f=4,
        dim_dt=4,
        dim_in_proj=8,
        dim_patch_embed=8,
        dim_x_proj=4,
        dim_x_proj_f=4,
        encoder_hidden_size=None,
        learnable_A=True,
        learnable_A_v2=True,
        learnable_D=False,
        learnable_D_v2=False,
        learnable_cls_token=True,
        learnable_cls_token_v2=True,
        learnable_conv1d=False,
        learnable_conv1d_v2=True,
        learnable_pos_embed=False,
        learnable_pos_embed_v2=True,
        lora_B=True,
        lora_C=False,
        lora_X=True,
        lora_Z=False,
        lora_d=False,
        lora_dt=True,
        lora_in_proj=True,
        lora_out_proj=True,
        lora_patch_embed=False,
        lora_x_proj=False,
        num_virtual_tokens=1,
        prefix_projection=True,
        prefix_tuning=True,
        prefix_type='inner_dual_prefix',
        prompt_num_tokens=1,
        prompt_projection=True,
        prompt_tuning=False,
        prompt_type='prefix_suffix',
        s=0.1,
        s_B=0.1,
        s_C=0.1,
        s_X=0.1,
        s_Z=0.1,
        s_d=0.1,
        s_dt=0.1,
        s_in_proj=0.1,
        s_patch_embed=1,
        s_x_proj=0.1,
        scan_A_constant=None,
        scan_A_copy_from_last=True,
        scan_addition_num=1,
        scan_addition_pos='prefix',
        train_A=True,
        train_D=True,
        train_additional_scan=True,
        train_cls_token=True,
        train_conv1d=True,
        train_lora_B=True,
        train_lora_C=True,
        train_lora_X=True,
        train_lora_Z=True,
        train_lora_d=True,
        train_lora_dt=True,
        train_lora_in_proj=True,
        train_lora_out_proj=True,
        train_lora_patch_embed=True,
        train_lora_x_proj=True,
        train_pos_embed=True,
        train_prefix=True,
        train_prompt=True,
        zero_init_x_proj=False,
        module=dict(
            type='VisionMamba',
            patch_size=16, embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, 
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
        in_channels=384,
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
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0),
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.dist_token': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0),
            '.A_log': dict(decay_mult=0.0),
            '.A_b_log': dict(decay_mult=0.0),
            '.D.': dict(decay_mult=0.0),
            '.D_b.': dict(decay_mult=0.0),
            'lora_out_proj': dict(lr_mult=1., decay_mult=1.),
            'lora_in_proj': dict(lr_mult=1., decay_mult=1.),
            'lora_X': dict(lr_mult=1., decay_mult=1.),
            'lora_Z': dict(lr_mult=1., decay_mult=1.),
            'lora_x_proj': dict(lr_mult=0.1, decay_mult=1.),
            'lora_d': dict(lr_mult=1., decay_mult=1.),
            'lora_B': dict(lr_mult=1., decay_mult=1.),
            'lora_C': dict(lr_mult=1., decay_mult=1.),
            'lora_dt': dict(lr_mult=1., decay_mult=1.),
            'lora_patch_embed': dict(lr_mult=1., decay_mult=1.),
            'prefix_encoder': dict(lr_mult=1.0, decay_mult=1.),
            'prompt_encoder': dict(lr_mult=1., decay_mult=1.),
            'learnable_A': dict(lr_mult=1.0, decay_mult=100.0),
            'learnable_D': dict(lr_mult=1.0, decay_mult=100.0),
            'learnable_conv1d': dict(lr_mult=1.0, decay_mult=100.0),
            'learnable_patch_embed': dict(lr_mult=1.0, decay_mult=100.0),
            'learnable_cls_token': dict(lr_mult=1.0, decay_mult=100.0),
            'A_b_log_scan_addi': dict(lr_mult=1., decay_mult=0.),
            'A_log_scan_addi': dict(lr_mult=1., decay_mult=0.),
            'x_proj_scan_addi': dict(lr_mult=1., decay_mult=1.),
            'x_proj_b_scan_addi': dict(lr_mult=1., decay_mult=1.),
            
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