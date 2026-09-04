# RecycleLoRA (DINOv2-L + Mask2Former), synthetic multi-source:
# GTAV (12,403 random subset) + Synthia -> {Cityscapes, BDD, Mapillary}
_base_ = [
    "../_base_/datasets/dg_gs2cbm_512x512.py",
    "../_base_/default_runtime.py",
    "../_base_/models/recyclelora_dinov2_mask2former.py",
]

train_dataloader = dict(batch_size=4)

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
backbone_lr_mult = dict(lr_mult=0.5, decay_mult=1.0)
lora_main_mult = dict(lr_mult=1.0, decay_mult=0.0)
lora_sub_mult = dict(lr_mult=0.5, decay_mult=0.0)

optim_wrapper = dict(
    constructor="PEFTOptimWrapperConstructor",
    optimizer=dict(
        type="AdamW", lr=0.0001, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": backbone_lr_mult,
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "lora_A_main": lora_main_mult,
            "lora_B_main": lora_main_mult,
            "lora_A_sub": lora_sub_mult,
            "lora_B_sub": lora_sub_mult,
        },
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False),
    dict(
        type="CosineAnnealingParamScheduler",
        param_name="weight_decay",
        eta_min=0.00001,
        by_epoch=False,
        begin=0,
        end=40000,
    ),
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=4000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=4000, max_keep_ckpts=3
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)
custom_hooks = []
