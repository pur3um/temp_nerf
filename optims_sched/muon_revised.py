# muon_revised.py
# Drop-in replacement for optims/muon.py with rank metadata for RankWSD-Auto.

import torch
import torch.distributed as dist


def _flattened_matrix_shape(p: torch.Tensor):
    if p.ndim == 4:
        return int(p.shape[0]), int(p.numel() // max(1, p.shape[0]))
    if p.ndim > 2:
        return int(p.shape[0]), int(p.numel() // max(1, p.shape[0]))
    if p.ndim == 2:
        return int(p.shape[0]), int(p.shape[1])
    return 1, int(p.numel())


def _infer_full_rank_summary(params):
    ranks = []
    for p in params:
        if p is None:
            continue
        m, n = _flattened_matrix_shape(p)
        ranks.append(max(1, min(m, n)))
    if not ranks:
        return {
            "full_rank_min": 1,
            "full_rank_max": 1,
            "full_rank_mean": 1.0,
            "full_rank_num_tensors": 0,
        }
    return {
        "full_rank_min": int(min(ranks)),
        "full_rank_max": int(max(ranks)),
        "full_rank_mean": float(sum(ranks) / len(ranks)),
        "full_rank_num_tensors": int(len(ranks)),
    }


def _attach_fullrank_muon_metadata(group):
    summary = _infer_full_rank_summary(group.get("params", []))
    full_rank = int(summary["full_rank_max"])
    group["use_muon"] = True
    group["rank_wsd_role"] = "full_rank_muon"
    group["is_full_rank"] = True
    group["is_progressive_rank"] = False
    group["rank_schedule_type"] = "fixed_full_rank"
    group["current_method"] = "muon_full_rank_ns"
    group["current_rank"] = full_rank
    group["current_target_rank"] = full_rank
    group["rank_start"] = full_rank
    group["rank_end"] = full_rank
    group["rank_saturated"] = True
    group["rank_progress"] = 1.0
    group["rank_wsd_decay_ready"] = True
    group["rank_saturation_step"] = 0
    group["full_rank_min"] = summary["full_rank_min"]
    group["full_rank_max"] = summary["full_rank_max"]
    group["full_rank_mean"] = summary["full_rank_mean"]
    group["full_rank_num_tensors"] = summary["full_rank_num_tensors"]
    return group


def _attach_aux_adam_metadata(group):
    group["rank_wsd_role"] = "aux_adam"
    group["is_full_rank"] = False
    group["is_progressive_rank"] = False
    group["rank_saturated"] = True
    group["rank_wsd_decay_ready"] = True
    return group


def _distributed_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _distributed_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _maybe_all_gather_param(param, gathered):
    if dist.is_available() and dist.is_initialized():
        dist.all_gather(gathered, param)


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration for Muon orthogonalization.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    elif update.ndim > 2:
        update = update.view(update.shape[0], -1)
    elif update.ndim < 2:
        update = update.view(1, -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """Distributed Muon with full-rank metadata for RankWSD-Auto."""

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)
        for group in self.param_groups:
            _attach_fullrank_muon_metadata(group)
            group["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = _distributed_world_size()
        rank = _distributed_rank()
        for group in self.param_groups:
            group["step"] = int(group.get("step", 0)) + 1
            _attach_fullrank_muon_metadata(group)
            params = group["params"]
            if not params:
                continue
            params_pad = params + [torch.empty_like(params[-1])] * ((world_size - len(params) % world_size) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=int(group.get("ns_steps", 5)),
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                if world_size > 1:
                    _maybe_all_gather_param(
                        params_pad[base_i + rank],
                        params_pad[base_i:base_i + world_size],
                    )
        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """Single-device Muon with full-rank metadata for RankWSD-Auto."""

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            _attach_fullrank_muon_metadata(group)
            group["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] = int(group.get("step", 0)) + 1
            _attach_fullrank_muon_metadata(group)
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=int(group.get("ns_steps", 5)),
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Distributed Muon + AuxAdam with RankWSD-Auto metadata."""

    def __init__(self, param_groups):
        normalized_groups = []
        for group in param_groups:
            assert "use_muon" in group
            g = dict(group)
            if g["use_muon"]:
                g["params"] = sorted(g["params"], key=lambda x: x.size(), reverse=True)
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
                _attach_fullrank_muon_metadata(g)
                g.setdefault("step", 0)
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.0)
                _attach_aux_adam_metadata(g)
            normalized_groups.append(g)
        super().__init__(normalized_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = _distributed_world_size()
        rank = _distributed_rank()
        for group in self.param_groups:
            if group["use_muon"]:
                group["step"] = int(group.get("step", 0)) + 1
                _attach_fullrank_muon_metadata(group)
                params = group["params"]
                if not params:
                    continue
                params_pad = params + [torch.empty_like(params[-1])] * ((world_size - len(params) % world_size) % world_size)
                for base_i in range(0, len(params), world_size):
                    if base_i + rank < len(params):
                        p = params[base_i + rank]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            ns_steps=int(group.get("ns_steps", 5)),
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    if world_size > 1:
                        _maybe_all_gather_param(
                            params_pad[base_i + rank],
                            params_pad[base_i:base_i + world_size],
                        )
            else:
                _attach_aux_adam_metadata(group)
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Single-device Muon + AuxAdam with RankWSD-Auto metadata."""

    def __init__(self, param_groups):
        normalized_groups = []
        for group in param_groups:
            assert "use_muon" in group
            g = dict(group)
            if g["use_muon"]:
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("weight_decay", 0.0)
                g.setdefault("ns_steps", 5)
                _attach_fullrank_muon_metadata(g)
                g.setdefault("step", 0)
            else:
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-10)
                g.setdefault("weight_decay", 0.0)
                _attach_aux_adam_metadata(g)
            normalized_groups.append(g)
        super().__init__(normalized_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                group["step"] = int(group.get("step", 0)) + 1
                _attach_fullrank_muon_metadata(group)
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=int(group.get("ns_steps", 5)),
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                _attach_aux_adam_metadata(group)
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss
