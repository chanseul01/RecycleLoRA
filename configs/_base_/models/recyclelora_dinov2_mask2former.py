crop_size  = (512, 512)
num_classes = 19

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=crop_size,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    ),

    # ───────────────────────────── backbone ────────────────────────────
    backbone=dict(
        type          = "RecycleLoRADinoVisionTransformer",
        img_size      = 512,
        patch_size    = 16,
        embed_dim     = 1024,
        depth         = 24,
        num_heads     = 16,
        out_indices   = (7, 11, 15, 23),

        mlp_ratio     = 4,
        ffn_layer     = "mlp",
        init_values   = 1e-5,
        block_chunks  = 0,
        qkv_bias      = True,
        proj_bias     = True,
        ffn_bias      = True,

        r_max         = 36,          # total rank (main 32 + sub 4)
        r_main        = 32,
        lora_alpha    = 1.0,         # forward multiplier = lora_alpha / r_max
        n_skip_layers = 0,           # number of leading blocks left unadapted
        bias_tune     = False,
        merge_weights = True,

        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2_converted.pth"
        ),
    ),

    # ───────────────────────────── decode_head ─────────────────────────
    decode_head=dict(
        type="Mask2FormerHead",
        in_channels=[1024, 1024, 1024, 1024],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=100,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type="mmdet.MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256, num_heads=8, num_levels=3,
                        num_points=4, im2col_step=64, dropout=0.0,
                        batch_first=True, norm_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=1024,
                        num_fcs=2, ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True)),
                ),
            ),
            positional_encoding=dict(num_feats=128, normalize=True),
        ),
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True, num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256, num_heads=8,
                    attn_drop=0.0, proj_drop=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256, num_heads=8,
                    attn_drop=0.0, proj_drop=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256, feedforward_channels=2048,
                    num_fcs=2, act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0, add_identity=True),
            ),
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False, loss_weight=2.0,
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True, loss_weight=5.0),
        loss_dice=dict(
            type="mmdet.DiceLoss",
            use_sigmoid=True, activate=True,
            naive_dice=True, eps=1.0, loss_weight=5.0),
        train_cfg=dict(
            num_points=12544, oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="mmdet.HungarianAssigner",
                match_costs=[
                    dict(type="mmdet.ClassificationCost",  weight=2.0),
                    dict(type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True),
                    dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0),
                ]),
            sampler=dict(type="mmdet.MaskPseudoSampler"),
        ),
    ),

    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=crop_size, stride=(341, 341)),
)