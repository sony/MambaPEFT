
search_setting = {
    "model.backbone.lora_out_proj": dict(categories=[True, False]),
    # "model.backbone.dim": dict(int_min=8, int_max=64, step=8),
    # "model.backbone.s": dict(categories=[0.01, 0.1, 1.]),
    
    "model.backbone.lora_in_proj": dict(categories=[True, False]),
    # "model.backbone.dim_in_proj": dict(int_min=8, int_max=64, step=8),
    # "model.backbone.s_in_proj": dict(categories=[0.01, 0.1, 1.]),

    "model.backbone.lora_X": dict(categories=[True, False]),
    # "model.backbone.dim_X": dict(int_min=8, int_max=64, step=8),
    # "model.backbone.s_X": dict(categories=[0.01, 0.1, 1.]),

    "model.backbone.lora_Z": dict(categories=[True, False]),
    # "model.backbone.dim_Z": dict(int_min=8, int_max=64, step=8),
    # "model.backbone.s_Z": dict(categories=[0.01, 0.1, 1.]),

    "model.backbone.lora_x_proj": dict(categories=[True, False]),
    # "model.backbone.dim_x_proj": dict(int_min=4, int_max=32, step=4),
    # "model.backbone.s_x_proj": dict(categories=[0.01, 0.1, 1.]),

    "model.backbone.lora_d": dict(categories=[True, False]),
    # "model.backbone.dim_d": dict(int_min=4, int_max=16, step=4),
    # "model.backbone.s_d": dict(categories=[0.01, 0.1, 1.]),
    
    "model.backbone.lora_B": dict(categories=[True, False]),
    # "model.backbone.dim_B": dict(int_min=4, int_max=16, step=4),
    # "model.backbone.s_B": dict(categories=[0.01, 0.1, 1.]),
    
    "model.backbone.lora_C": dict(categories=[True, False]),
    # "model.backbone.dim_C": dict(int_min=4, int_max=16, step=4),
    # "model.backbone.s_C": dict(categories=[0.01, 0.1, 1.]),

    "model.backbone.lora_dt": dict(categories=[True, False]),
    # "model.backbone.dim_dt": dict(int_min=8, int_max=64, step=8),
    # "model.backbone.s_dt": dict(categories=[0.01, 0.1, 1.]),

    "model.backbone.lora_patch_embed": dict(categories=[True, False]),
    
    "model.backbone.prefix_tuning": dict(categories=[True, False]),
    # "model.backbone.prefix_type": dict(categories=["inner_single_prefix", "inner_single_infix"]),
    # "model.backbone.num_virtual_tokens": dict(int_min=1, int_max=4, step=1),
    
    "model.backbone.prompt_tuning": dict(categories=[True, False]),

    "model.backbone.additional_scan": dict(categories=[True, False]),
    # "model.backbone.scan_addition_pos": dict(categories=["suffix"]),
    # "model.backbone.scan_addition_num": dict(int_min=1, int_max=8, step=1),

    "model.backbone.learnable_A": dict(categories=[True, False]),
    "model.backbone.learnable_D": dict(categories=[True, False]),
    "model.backbone.learnable_conv1d": dict(categories=[True, False]),
    "model.backbone.learnable_cls_token": dict(categories=[True, False]),
    "model.backbone.learnable_pos_embed": dict(categories=[True, False]),
    "model.backbone.learnable_bias": dict(categories=[True, False]),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_out_proj.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_out_proj.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_in_proj.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_in_proj.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_X.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_X.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_Z.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_Z.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_x_proj.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_x_proj.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_d.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_d.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_B.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_B.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_dt.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_dt.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),

    # "optim_wrapper.paramwise_cfg.custom_keys.prefix_encoder.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys.prefix_encoder.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),

    # "optim_wrapper.paramwise_cfg.custom_keys._scan_addi.lr_mult": dict(categories=[0.1, 0.5, 1., 5.]),
    # "optim_wrapper.paramwise_cfg.custom_keys._scan_addi.decay_mult": dict(categories=[0.1, 0.5, 1., 5.]),
}

metric_prefix = "accuracy/top1:"

