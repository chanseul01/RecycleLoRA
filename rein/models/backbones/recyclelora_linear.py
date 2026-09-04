import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg


def transpose(weight, fan_in_fan_out):
    if fan_in_fan_out:
        return weight.T
    return weight


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class RecycleLoRALinear(nn.Linear, LoraLayer):
    """RecycleLoRA dual-adapter linear layer.

    ``W = W_res + scaling * (B_main @ A_main + B_sub @ A_sub)``, with the main and
    sub adapters initialised from the minor and major RRQR directions of the
    pre-trained weight and the residual ``W_res`` frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        r_main: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)

        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out

        # Rank split: main / sub
        self.r_main = min(r_main, r) if r > 0 else 0
        self.r_sub = r - self.r_main if r > 0 else 0

        if r > 0:
            # Main adapter parameters
            if self.r_main > 0:
                self.lora_A_main = nn.Parameter(torch.zeros(self.r_main, in_features))
                self.lora_B_main = nn.Parameter(torch.zeros(out_features, self.r_main))

            # Sub adapter parameters
            if self.r_sub > 0:
                self.lora_A_sub = nn.Parameter(torch.zeros(self.r_sub, in_features))
                self.lora_B_sub = nn.Parameter(torch.zeros(out_features, self.r_sub))

            # Forward multiplier (lora_alpha / r).
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def initialize_and_subtract(self, pretrained_weight, pretrained_bias=None):
        """Initialise the adapters from ``pretrained_weight`` via RRQR and write
        the frozen residual ``W_res = W0 - scaling * (B@A)``."""
        with torch.no_grad():
            if self.fan_in_fan_out:
                weight = pretrained_weight.T.clone().float()
            else:
                weight = pretrained_weight.clone().float()

            if pretrained_bias is not None and self.bias is not None:
                self.bias.data.copy_(pretrained_bias)

            # Main adapter <- RRQR minor directions
            if hasattr(self, "lora_A_main") and self.r_main > 0:
                A_main, B_main = self.compute_rrqr_components(
                    weight, self.r_main, use_top=False
                )
                self.lora_A_main.data.copy_(A_main)
                self.lora_B_main.data.copy_(B_main)

            # Sub adapter <- RRQR major directions
            if hasattr(self, "lora_A_sub") and self.r_sub > 0:
                A_sub, B_sub = self.compute_rrqr_components(
                    weight, self.r_sub, use_top=True
                )
                self.lora_A_sub.data.copy_(A_sub)
                self.lora_B_sub.data.copy_(B_sub)

            # Initial adapter contribution
            delta_w = 0
            if self.r_main > 0:
                delta_w = delta_w + self.lora_B_main @ self.lora_A_main
            if self.r_sub > 0:
                delta_w = delta_w + self.lora_B_sub @ self.lora_A_sub

            # W_res = W0 - scaling * (B@A)
            residual = weight - delta_w * self.scaling
            if self.fan_in_fan_out:
                residual = residual.T
            self.weight.data.copy_(residual.to(self.weight.dtype))

    def rrqr_decomposition(self, weight):
        """Rank-Revealing (column-pivoted) QR decomposition: ``W P = Q R``."""
        weight_np = weight.detach().cpu().numpy()
        Q, R, P = linalg.qr(weight_np, pivoting=True, mode="economic")
        return Q, R, P

    def compute_rrqr_components(self, weight, r, use_top=False):
        """Build a low-rank adapter (B, A) from RRQR of ``weight``.

        ``use_top=True`` selects the top-r (major) directions, ``False`` the
        bottom-r (minor) directions.
        """
        if r <= 0:
            return None, None

        Q, R, P = self.rrqr_decomposition(weight)
        num_cols = Q.shape[1]
        r = min(r, num_cols)

        if use_top:
            # Most important r directions (sub adapter)
            Q_selected = Q[:, :r]
            selected_cols = [P[i] for i in range(r)]
        else:
            # Least important r directions (main adapter)
            Q_selected = Q[:, num_cols - r:]
            selected_cols = [P[num_cols - r + i] for i in range(r)]

        # B <- selected orthonormal columns of Q
        B = Q_selected

        # A <- sparse 0/1 selection matrix
        A = np.zeros((r, weight.shape[1]), dtype=np.float32)
        for i in range(r):
            A[i, selected_cols[i]] = 1.0

        A_tensor = torch.from_numpy(A).to(weight.device).float()
        B_tensor = torch.from_numpy(np.ascontiguousarray(B)).to(weight.device).float()
        return A_tensor, B_tensor

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if not mode and self.merge_weights and not self.merged:
            # Merge adapters for inference
            if self.r > 0:
                delta_w = 0
                if self.r_main > 0:
                    delta_w = delta_w + self.lora_B_main @ self.lora_A_main
                if self.r_sub > 0:
                    delta_w = delta_w + self.lora_B_sub @ self.lora_A_sub
                self.weight.data += transpose(delta_w, self.fan_in_fan_out) * self.scaling
            self.merged = True
        elif self.merge_weights and self.merged:
            # Un-merge adapters
            if self.r > 0:
                delta_w = 0
                if self.r_main > 0:
                    delta_w = delta_w + self.lora_B_main @ self.lora_A_main
                if self.r_sub > 0:
                    delta_w = delta_w + self.lora_B_sub @ self.lora_A_sub
                self.weight.data -= transpose(delta_w, self.fan_in_fan_out) * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            if self.r > 0 and self.merged:
                delta_w = 0
                if self.r_main > 0:
                    delta_w = delta_w + self.lora_B_main @ self.lora_A_main
                if self.r_sub > 0:
                    delta_w = delta_w + self.lora_B_sub @ self.lora_A_sub
                self.weight.data -= transpose(
                    delta_w.to(previous_dtype), self.fan_in_fan_out
                ) * self.scaling
                self.merged = False
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r > 0 and not self.merged:
            # Frozen residual W_res
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            lora_output = self.lora_dropout(x)
            if self.r_main > 0:
                main_output = lora_output @ self.lora_A_main.T
                main_output = main_output @ self.lora_B_main.T
                result += main_output * self.scaling
            if self.r_sub > 0:
                sub_output = lora_output @ self.lora_A_sub.T
                sub_output = sub_output @ self.lora_B_sub.T
                result += sub_output * self.scaling
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)
        return result

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, r_main={self.r_main}, r_sub={self.r_sub}, "
            f"lora_alpha={self.lora_alpha}, scaling={self.scaling}"
        )
