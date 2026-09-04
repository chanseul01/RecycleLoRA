# RecycleLoRA backbone: DINOv2 with RRQR-based dual (main/sub) LoRA adapters.
import torch.nn as nn
from mmseg.models.builder import BACKBONES

from .dino_v2 import DinoVisionTransformer
from .recyclelora_linear import RecycleLoRALinear
from .utils import set_requires_grad


def replace_linear_layers(
    module,
    lora_linear_class,
    r_max,
    r_main,
    lora_alpha,
    n_skip_layers,
    skip_names,
    merge_weights=True,
):
    """Replace every ``nn.Linear`` (outside ``skip_names`` and the first
    ``n_skip_layers * 4`` layers) with a RecycleLoRA adapter layer."""
    layer_count = 0
    for name, child in module.named_children():
        if skip_names and name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            layer_count += 1
            if layer_count > (n_skip_layers * 4):
                new_layer = lora_linear_class(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    r=r_max,
                    r_main=r_main,
                    lora_alpha=lora_alpha,
                    bias=(child.bias is not None),
                    merge_weights=merge_weights,
                )
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)
            else:
                # Freeze the first n_skip_layers blocks entirely
                child.weight.requires_grad = False
                if child.bias is not None:
                    child.bias.requires_grad = False
        else:
            replace_linear_layers(
                child,
                lora_linear_class,
                r_max,
                r_main,
                lora_alpha,
                n_skip_layers,
                skip_names,
                merge_weights,
            )


@BACKBONES.register_module()
class RecycleLoRADinoVisionTransformer(DinoVisionTransformer):
    """DINOv2 backbone with RecycleLoRA dual (main/sub) adapters.

    RRQR runs in ``init_weights``, after the pre-trained checkpoint is loaded.
    """

    def __init__(
        self,
        r_max: int = 16,
        r_main: int = 8,
        lora_alpha: float = 1.0,
        n_skip_layers: int = 8,
        bias_tune: bool = False,
        skip_names=None,
        merge_weights: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.r_max = r_max
        self.r_main = r_main
        self.lora_alpha = lora_alpha
        self.n_skip_layers = n_skip_layers
        self.bias_tune = bias_tune
        self.skip_names = skip_names or ["head"]
        self.merge_weights = merge_weights
        self._rrqr_initialized = False

        # Enable training of mask_token if present
        if hasattr(self, "mask_token"):
            self.mask_token.requires_grad_(True)

        # Swap Linear -> LoRA layers
        replace_linear_layers(
            self,
            RecycleLoRALinear,
            self.r_max,
            self.r_main,
            self.lora_alpha,
            self.n_skip_layers,
            self.skip_names,
            self.merge_weights,
        )

    def init_weights(self):
        # Load pre-trained DINOv2 weights, then run RRQR on them (once).
        super().init_weights()
        if not self._rrqr_initialized:
            for module in self.modules():
                if isinstance(module, RecycleLoRALinear):
                    module.initialize_and_subtract(
                        module.weight.data,
                        module.bias.data if module.bias is not None else None,
                    )
            self._rrqr_initialized = True

    def forward_features(self, x, masks=None):
        return super().forward_features(x, masks)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            requires = []
            # Adapter parameters are trainable
            for name, module in self.named_modules():
                if isinstance(module, RecycleLoRALinear) and name:
                    if hasattr(module, "lora_A_main"):
                        requires.append(f"{name}.lora_A_main")
                        requires.append(f"{name}.lora_B_main")
                    if hasattr(module, "lora_A_sub"):
                        requires.append(f"{name}.lora_A_sub")
                        requires.append(f"{name}.lora_B_sub")

            # Optionally train biases
            if self.bias_tune:
                for i in range(self.n_skip_layers, self.n_blocks):
                    requires += [
                        f"{i}.attn.qkv.bias",
                        f"{i}.attn.proj.bias",
                        f"{i}.mlp.fc1.bias",
                        f"{i}.mlp.fc2.bias",
                    ]

            if hasattr(self, "mask_token"):
                requires.append("mask_token")

            set_requires_grad(self, requires)
        return self
