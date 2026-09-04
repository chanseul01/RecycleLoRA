# Pretrained backbone checkpoints

The configs expect the converted DINOv2 backbone at
`checkpoints/dinov2_converted.pth`. It is **not** committed to the repo
(it is large and downloadable). Prepare it as follows:

1. Download the official DINOv2-Large (with registers is not required) weights,
   e.g. `dinov2_vitl14_pretrain.pth`, from
   <https://github.com/facebookresearch/dinov2>.

2. Convert them to the 16-patch, 512-resolution layout used here:

   ```bash
   python tools/convert_models/convert_dinov2.py \
       /path/to/dinov2_vitl14_pretrain.pth \
       checkpoints/dinov2_converted.pth \
       --kernel 16 --height 512 --width 512
   ```

For the EVA02-L backbone use
`tools/convert_models/convert_eva2_512x512.py` analogously.
