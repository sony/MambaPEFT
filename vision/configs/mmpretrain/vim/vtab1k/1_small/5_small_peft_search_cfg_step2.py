
search_setting = {
    "remove_method": dict(remove_categories=[
        "model.backbone.lora_out_proj",
        "model.backbone.lora_in_proj",
        "model.backbone.lora_X",
        "model.backbone.lora_B",
        "model.backbone.lora_dt",
        "model.backbone.prefix_tuning",
        "model.backbone.learnable_A",
        "model.backbone.learnable_cls_token",
        "_not_remove",
    ]),

    # "model.backbone.lora_out_proj": dict(categories=[True, False]),
    "model.backbone.dim": dict(int_min=4, int_max=16, step=4),
    "model.backbone.s": dict(min=0.01, max=1., log=True),
    
    # #"model.backbone.lora_in_proj": dict(categories=[True, False]),
    "model.backbone.dim_in_proj": dict(int_min=4, int_max=16, step=4),
    "model.backbone.s_in_proj": dict(min=0.01, max=1., log=True),

    # #"model.backbone.lora_X": dict(categories=[True, False]),
    "model.backbone.dim_X": dict(int_min=4, int_max=16, step=4),
    "model.backbone.s_X": dict(min=0.01, max=1., log=True),

    # #"model.backbone.lora_Z": dict(categories=[True, False]),
    # "model.backbone.dim_Z": dict(int_min=4, int_max=64, step=4),
    # "model.backbone.s_Z": dict(min=0.01, max=1., log=True),

    # #"model.backbone.lora_x_proj": dict(categories=[True, False]),
    # "model.backbone.dim_x_proj": dict(int_min=4, int_max=48, step=4),
    # "model.backbone.s_x_proj": dict(min=0.01, max=1., log=True),

    # #"model.backbone.lora_d": dict(categories=[True, False]),
    # "model.backbone.dim_d": dict(int_min=4, int_max=32, step=4),
    # "model.backbone.s_d": dict(min=0.01, max=1., log=True),
    
    #"model.backbone.lora_B": dict(categories=[True, False]),
    "model.backbone.dim_B": dict(int_min=4, int_max=12, step=4),
    "model.backbone.s_B": dict(min=0.01, max=1., log=True),
    
    # #"model.backbone.lora_C": dict(categories=[True, False]),
    # "model.backbone.dim_C": dict(int_min=4, int_max=32, step=4),
    # "model.backbone.s_C": dict(min=0.01, max=1., log=True),

    #"model.backbone.lora_dt": dict(categories=[True, False]),
    "model.backbone.dim_dt": dict(int_min=4, int_max=12, step=4),
    "model.backbone.s_dt": dict(min=0.01, max=1., log=True),

    # #"model.backbone.lora_patch_embed": dict(categories=[True, False]),
    # "model.backbone.dim_patch_embed": dict(int_min=4, int_max=64, step=4),
    # "model.backbone.s_patch_embed": dict(min=0.01, max=1., log=True),
    
    #"model.backbone.prefix_tuning": dict(categories=[True, False]),
    # "model.backbone.prefix_type": dict(categories=["inner_single_prefix", "inner_single_infix"]),
    "model.backbone.num_virtual_tokens": dict(int_min=1, int_max=3, step=1),

    
    #"model.backbone.prompt_tuning": dict(categories=[True, False]),
    #"model.backbone.prompt_num_tokens": dict(int_min=1, int_max=6, step=1),

    #"model.backbone.additional_scan": dict(categories=[True, False]),
    # "model.backbone.scan_addition_num": dict(int_min=1, int_max=6, step=1),

    # "model.backbone.learnable_A": dict(categories=[True, False]),
    # "model.backbone.learnable_D": dict(categories=[True, False]),
    # "model.backbone.learnable_conv1d": dict(categories=[True, False]),
    # "model.backbone.learnable_cls_token": dict(categories=[True, False]),
    # "model.backbone.learnable_pos_embed": dict(categories=[True, False]),

    "optim_wrapper.paramwise_cfg.custom_keys.lora_out_proj.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.lora_out_proj.decay_mult": dict(min=0.1, max=10., log=True),

    "optim_wrapper.paramwise_cfg.custom_keys.lora_in_proj.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.lora_in_proj.decay_mult": dict(min=0.1, max=10., log=True),
    
    "optim_wrapper.paramwise_cfg.custom_keys.lora_X.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.lora_X.decay_mult": dict(min=0.1, max=10., log=True),
    
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_Z.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_Z.decay_mult": dict(min=0.01, max=10., log=True),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_x_proj.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_x_proj.decay_mult": dict(min=0.01, max=10., log=True),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_d.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_d.decay_mult": dict(min=0.01, max=10., log=True),

    "optim_wrapper.paramwise_cfg.custom_keys.lora_B.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.lora_B.decay_mult": dict(min=0.1, max=10., log=True),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_C.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_C.decay_mult": dict(min=0.01, max=10., log=True),

    "optim_wrapper.paramwise_cfg.custom_keys.lora_dt.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.lora_dt.decay_mult": dict(min=0.1, max=10., log=True),

    # "optim_wrapper.paramwise_cfg.custom_keys.lora_patch_embed.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.lora_patch_embed.decay_mult": dict(min=0.01, max=10., log=True),

    "optim_wrapper.paramwise_cfg.custom_keys.prefix_encoder.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.prefix_encoder.decay_mult": dict(min=0.1, max=10., log=True),

    # "optim_wrapper.paramwise_cfg.custom_keys.prompt_encoder.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.prompt_encoder.decay_mult": dict(min=0.01, max=10., log=True),

    "optim_wrapper.paramwise_cfg.custom_keys.learnable_A.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.learnable_A.decay_mult": dict(min=0.1, max=100., log=True),

    # "optim_wrapper.paramwise_cfg.custom_keys.learnable_D.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.learnable_D.decay_mult": dict(min=0.1, max=100., log=True),

    # "optim_wrapper.paramwise_cfg.custom_keys.learnable_conv1d.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.learnable_conv1d.decay_mult": dict(min=0.1, max=100., log=True),
    
    # "optim_wrapper.paramwise_cfg.custom_keys.learnable_patch_embed.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.learnable_patch_embed.decay_mult": dict(min=0.1, max=100., log=True),

    "optim_wrapper.paramwise_cfg.custom_keys.learnable_cls_token.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.learnable_cls_token.decay_mult": dict(min=0.1, max=100., log=True),
    
    # "optim_wrapper.paramwise_cfg.custom_keys.A_log_scan_addi.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.A_b_log_scan_addi.lr_mult": dict(origin="optim_wrapper.paramwise_cfg.custom_keys.A_log_scan_addi.lr_mult"),
    # "optim_wrapper.paramwise_cfg.custom_keys.A_log_scan_addi.decay_mult": dict(min=0.01, max=10., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.A_b_log_scan_addi.decay_mult": dict(origin="optim_wrapper.paramwise_cfg.custom_keys.A_log_scan_addi.decay_mult"),

    # "optim_wrapper.paramwise_cfg.custom_keys.x_proj_scan_addi.lr_mult": dict(min=0.1, max=5., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.x_proj_b_scan_addi.lr_mult": dict(origin="optim_wrapper.paramwise_cfg.custom_keys.x_proj_scan_addi.lr_mult"),
    # "optim_wrapper.paramwise_cfg.custom_keys.x_proj_scan_addi.decay_mult": dict(min=0.01, max=10., log=True),
    # "optim_wrapper.paramwise_cfg.custom_keys.x_proj_b_scan_addi.decay_mult": dict(origin="optim_wrapper.paramwise_cfg.custom_keys.x_proj_scan_addi.decay_mult"),

    "optim_wrapper.paramwise_cfg.custom_keys.head.lr_mult": dict(min=0.1, max=5., log=True),
    "optim_wrapper.paramwise_cfg.custom_keys.head.decay_mult": dict(min=0.1, max=10., log=True),
}

metric_prefix = "accuracy/top1:"

