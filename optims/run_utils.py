import os
import math
import torch
from typing import Optional, Tuple

# =============================================================================
# Muon-like optimizer helpers
# =============================================================================
MUON_LIKE_FIRST_LAYER_MODELS = {'relu_ffn', 'gauss_ffn', 'relu_pos_enc'}

def muon_like_includes_first_layer(
        model_name: str,
        optimize_first_layer: bool
    ) -> bool:
    return bool(optimize_first_layer and model_name in MUON_LIKE_FIRST_LAYER_MODELS)


def split_muon_like_named_params(
        model,
        model_name: str,
        optimize_first_layer: bool = False
    ):
    hidden_named_params = []
    other_named_params = []
    include_first_layer = muon_like_includes_first_layer(model_name, optimize_first_layer)

    if hasattr(model, 'mlp') and isinstance(model.mlp, torch.nn.Sequential):
        num_mlp_layers = len(model.mlp)

        for name, param in model.named_parameters():
            is_target = False
            if 'mlp' in name and 'weight' in name and param.ndim >= 2:
                try:
                    layer_idx = int(name.split('.')[1])
                    is_hidden_layer = 0 < layer_idx < num_mlp_layers - 1
                    is_first_layer = include_first_layer and layer_idx == 0
                    is_target = is_hidden_layer or is_first_layer
                except (ValueError, IndexError):
                    pass

            if is_target:
                hidden_named_params.append((name, param))
            else:
                other_named_params.append((name, param))
    else:
        other_named_params = list(model.named_parameters())

    return hidden_named_params, other_named_params


def parse_pair(param_str: str) -> Tuple[float, float]:
    values = [float(x.strip()) for x in param_str.split(',') if x.strip()]
    # import pdb; pdb.set_trace()
    # param_str_list = str(param_str)
    # values = [float(x.strip()) for x in param_str_list.split(',') if x.strip()]
    if len(values) != 2:
        raise ValueError(f'Expected exactly two comma-separated values, got: {param_str}')
    return values[0], values[1]

def get_monotonic_rank(step: int, start_rank: int, end_rank: int, warmup_steps: int) -> int:
    """
    Monotonic non-decreasing linear schedule.

    step <= 1             -> start_rank
    step >= warmup_steps  -> end_rank
    otherwise             -> linear interpolation with integer flooring
    """
    step = max(1, int(step))
    start_rank = int(start_rank)
    end_rank = int(end_rank)
    warmup_steps = int(warmup_steps)

    if end_rank <= start_rank:
        return start_rank
    if warmup_steps <= 1:
        return end_rank
    if step >= warmup_steps:
        return end_rank

    rank = start_rank + ((end_rank - start_rank) * (step - 1)) // (warmup_steps - 1)
    return max(start_rank, min(rank, end_rank))


def get_cosine_rank(step: int, start_rank: int, end_rank: int, warmup_steps: int) -> int:
    """
    Monotonic non-decreasing cosine-ease-out schedule.

    step <= 1             -> start_rank
    step >= warmup_steps  -> end_rank
    otherwise             -> cosine-like fast early increase, then slow saturation
    """
    step = max(1, int(step))
    start_rank = int(start_rank)
    end_rank = int(end_rank)
    warmup_steps = int(warmup_steps)

    if end_rank <= start_rank:
        return start_rank
    if warmup_steps <= 1:
        return end_rank
    if step >= warmup_steps:
        return end_rank

    # normalized progress in [0, 1]
    t = (step - 1) / float(warmup_steps - 1)

    # ease-out cosine / equivalently sin(pi/2 * t)
    p = math.sin(0.5 * math.pi * t)

    rank = start_rank + int((end_rank - start_rank) * p)
    return max(start_rank, min(rank, end_rank))


def get_log_rank(
    step: int,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
    log_scale: float = 9.0,
) -> int:
    """
    Monotonic non-decreasing logarithmic schedule.

    step <= 1             -> start_rank
    step >= warmup_steps  -> end_rank
    otherwise             -> very fast early increase, then slowly saturates

    log_scale:
        Larger values make the early jump stronger.
    """
    step = max(1, int(step))
    start_rank = int(start_rank)
    end_rank = int(end_rank)
    warmup_steps = int(warmup_steps)

    if end_rank <= start_rank:
        return start_rank
    if warmup_steps <= 1:
        return end_rank
    if step >= warmup_steps:
        return end_rank

    t = (step - 1) / float(warmup_steps - 1)

    # normalized log curve in [0, 1]
    p = math.log1p(log_scale * t) / math.log1p(log_scale)

    rank = start_rank + int((end_rank - start_rank) * p)
    return max(start_rank, min(rank, end_rank))


def get_exponential_rank(
    step: int,
    start_rank: int,
    end_rank: int,
    warmup_steps: int,
    exp_scale: float = 5.0,
) -> int:
    """
    Monotonic non-decreasing exponential schedule.

    step <= 1             -> start_rank
    step >= warmup_steps  -> end_rank
    otherwise             -> slow early increase, then accelerates later

    exp_scale:
        Larger values make the late acceleration stronger.
    """
    step = max(1, int(step))
    start_rank = int(start_rank)
    end_rank = int(end_rank)
    warmup_steps = int(warmup_steps)

    if end_rank <= start_rank:
        return start_rank
    if warmup_steps <= 1:
        return end_rank
    if step >= warmup_steps:
        return end_rank

    t = (step - 1) / float(warmup_steps - 1)

    # normalized exponential curve in [0, 1]
    p = (math.exp(exp_scale * t) - 1.0) / (math.exp(exp_scale) - 1.0)

    rank = start_rank + int((end_rank - start_rank) * p)
    return max(start_rank, min(rank, end_rank))


def choose_rank_proxy(M, delta, rank_list):
    m, n = M.shape
    fro2_M = (M * M).sum()

    best = None
    for r in rank_list:
        Omega = torch.randn(n, r, device=M.device, dtype=M.dtype)
        Q, _ = torch.linalg.qr(M @ Omega, mode="reduced")
        B = Q.T @ M

        fro2_res = torch.clamp(fro2_M - (B * B).sum(), min=0.0)
        fro_res = torch.sqrt(fro2_res)

        # rank((I-QQ^T)M) <= min(m-r, n)
        upper = math.sqrt(max(1, min(m - r, n))) * fro_res

        if upper <= delta:
            return r, Q, B, upper
        best = (r, Q, B, upper)

    return best