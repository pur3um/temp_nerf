"""Safer MoFaSGD optimizer implementation for NeRF experiments.

This file is intended to replace the prototype optimizer implementation used in
MoFaSGD-on-NeRF experiments.  It keeps the public implementation's main idea
(rank-r momentum factors) but fixes the practical issues that make fair NeRF
comparisons brittle:

* State is initialized from the first gradient by default, not from the weight.
* The first gradient is not counted twice.
* The default update is U V^T with a separate near-zero threshold; the
  reciprocal-S variant has its own epsilon and a conservative clipping default.
* QR/SVD failures are counted, while CUDA OOM is re-raised instead of hidden.
* Requested rank is preserved up to min(m, n); tensors that cannot use the fast
  2r QR path transparently fall back to the explicit full update.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer


class MomentumFactorizedSGD(Optimizer):
    r"""Momentum Factorized SGD for matrix-shaped parameters.

    For each 2D parameter W, the optimizer stores a rank-r factorization

        M ~= U diag(S) V^T

    of the momentum-like matrix.  The parameter update is either

        eta1 * U V^T + eta2 * orthogonal_gradient

    by default, or the reciprocal-S variant

        eta1 * U diag(1 / S) V^T + eta2 * orthogonal_gradient

    when ``use_ones_for_nonzero_s=False``.

    Notes
    -----
    This optimizer intentionally supports only 2D dense parameters.  Use a
    second optimizer such as Adam for biases, scalar/vector parameters, and
    NeRF output heads.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        rank: int = 16,
        beta: float = 0.9,
        eta1: float = 1.0,
        eta2: float = 0.0,
        use_current_projection: bool = False,
        use_ones_for_nonzero_s: bool = True,
        eps: float = 1e-4,
        uv_eps: float = 0.0,
        nesterov: bool = False,
        max_value: float = 5.0,
        weight_decay: float = 0.0,
        init_from_grad: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must satisfy 0 <= beta < 1, got {beta}")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if uv_eps < 0.0:
            raise ValueError(f"uv_eps must be non-negative, got {uv_eps}")
        if max_value <= 0.0:
            raise ValueError(f"max_value must be positive, got {max_value}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        defaults = dict(
            lr=lr,
            rank=int(rank),
            beta=float(beta),
            eta1=float(eta1),
            eta2=float(eta2),
            use_current_projection=bool(use_current_projection),
            use_ones_for_nonzero_s=bool(use_ones_for_nonzero_s),
            eps=float(eps),
            uv_eps=float(uv_eps),
            nesterov=bool(nesterov),
            max_value=float(max_value),
            weight_decay=float(weight_decay),
            init_from_grad=bool(init_from_grad),
        )
        super().__init__(params, defaults)

    @staticmethod
    def _effective_rank(shape: torch.Size, requested_rank: int) -> int:
        if len(shape) != 2:
            raise RuntimeError(f"MomentumFactorizedSGD only supports 2D parameters, got {tuple(shape)}")
        m, n = int(shape[0]), int(shape[1])
        if m < 1 or n < 1:
            raise RuntimeError(f"Invalid 2D parameter shape: {tuple(shape)}")
        return max(1, min(int(requested_rank), m, n))

    @staticmethod
    def _work_dtype(dtype: torch.dtype) -> torch.dtype:
        if dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return dtype

    @staticmethod
    def _truncated_svd(x: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return U, S, V where x ~= U diag(S) V^T."""
        if x.dim() != 2:
            raise RuntimeError(f"SVD source must be 2D, got {tuple(x.shape)}")
        orig_dtype = x.dtype
        work_dtype = MomentumFactorizedSGD._work_dtype(orig_dtype)
        x_work = x.to(dtype=work_dtype)
        U, S, Vh = torch.linalg.svd(x_work, full_matrices=False)
        rank_eff = min(rank, S.numel())
        U = U[:, :rank_eff].contiguous().to(dtype=orig_dtype)
        S = S[:rank_eff].contiguous().to(dtype=orig_dtype)
        V = Vh[:rank_eff, :].transpose(0, 1).contiguous().to(dtype=orig_dtype)
        return U, S, V

    def _init_state(self, p: torch.nn.Parameter, group: Dict[str, Any]) -> None:
        if p.dim() != 2:
            raise RuntimeError("MomentumFactorizedSGD only supports 2D parameters.")

        requested_rank = int(group["rank"])
        rank = self._effective_rank(p.shape, requested_rank)
        source = p.grad.detach() if (group["init_from_grad"] and p.grad is not None) else p.detach()
        U, S, V = self._truncated_svd(source, rank)

        state = self.state[p]
        state["U"] = U.clone()
        state["S"] = S.clone()
        state["V"] = V.clone()
        state["rank"] = int(U.shape[1])
        state["requested_rank"] = requested_rank
        state["rank_clipped"] = bool(U.shape[1] != requested_rank)
        state["step"] = 0
        state["fallback_count"] = 0
        state["qr_failure_count"] = 0

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta = float(group["beta"])
            eta1 = float(group["eta1"])
            eta2 = float(group["eta2"])
            weight_decay = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("MomentumFactorizedSGD does not support sparse gradients.")
                if p.dim() != 2:
                    raise RuntimeError("MomentumFactorizedSGD only supports 2D parameters.")

                G = p.grad.detach()
                state = self.state[p]
                initialized_this_step = False
                if len(state) == 0:
                    self._init_state(p, group)
                    state = self.state[p]
                    initialized_this_step = True

                U_prev = state["U"]
                V_prev = state["V"]

                if initialized_this_step and group["init_from_grad"]:
                    # First-step fix: the first gradient has already defined M.
                    # Do not run the momentum update again with the same G.
                    U_next = state["U"]
                    S_next = state["S"]
                    V_next = state["V"]
                else:
                    U_next, S_next, V_next, used_fallback, qr_failed = self._update_momentum_factor(
                        state["U"], state["S"], state["V"], G, beta
                    )
                    if used_fallback:
                        state["fallback_count"] = int(state.get("fallback_count", 0)) + 1
                    if qr_failed:
                        state["qr_failure_count"] = int(state.get("qr_failure_count", 0)) + 1

                    if group["nesterov"]:
                        # Kept for compatibility, but this is a Nesterov-like heuristic rather
                        # than the standard lookahead-gradient formulation.
                        U_next, S_next, V_next, used_fallback, qr_failed = self._update_momentum_factor(
                            U_next, S_next, V_next, G, beta
                        )
                        if used_fallback:
                            state["fallback_count"] = int(state.get("fallback_count", 0)) + 1
                        if qr_failed:
                            state["qr_failure_count"] = int(state.get("qr_failure_count", 0)) + 1

                state["U"] = U_next
                state["S"] = S_next
                state["V"] = V_next
                state["rank"] = int(U_next.shape[1])
                state["step"] = int(state.get("step", 0)) + 1

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                update = self._make_update(
                    G=G,
                    U_prev=U_prev,
                    V_prev=V_prev,
                    U_next=U_next,
                    S_next=S_next,
                    V_next=V_next,
                    eta1=eta1,
                    eta2=eta2,
                    use_current_projection=bool(group["use_current_projection"]),
                    use_ones_for_nonzero_s=bool(group["use_ones_for_nonzero_s"]),
                    eps=float(group["eps"]),
                    uv_eps=float(group["uv_eps"]),
                    max_value=float(group["max_value"]),
                )
                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _make_update(
        G: torch.Tensor,
        U_prev: torch.Tensor,
        V_prev: torch.Tensor,
        U_next: torch.Tensor,
        S_next: torch.Tensor,
        V_next: torch.Tensor,
        eta1: float,
        eta2: float,
        use_current_projection: bool,
        use_ones_for_nonzero_s: bool,
        eps: float,
        uv_eps: float,
        max_value: float,
    ) -> torch.Tensor:
        scale = torch.zeros_like(S_next)
        if use_ones_for_nonzero_s:
            # Do not use the reciprocal-S epsilon here.  A large eps such as 1e-4
            # can silently zero useful low-rank directions in NeRF.  Only exact
            # zero, or values below uv_eps if requested, are suppressed.
            active = torch.isfinite(S_next) & (S_next.abs() > uv_eps)
            scale[active] = 1.0
        else:
            active = torch.isfinite(S_next) & (S_next.abs() > eps)
            scale[active] = torch.reciprocal(S_next[active])
            scale = torch.clamp(scale, max=max_value)

        low_rank_update = (U_next * scale.unsqueeze(0)) @ V_next.transpose(0, 1)
        update = eta1 * low_rank_update

        if eta2 != 0.0:
            U_proj, V_proj = (U_next, V_next) if use_current_projection else (U_prev, V_prev)
            left_ortho = G - U_proj @ (U_proj.transpose(0, 1) @ G)
            right_ortho = left_ortho - (left_ortho @ V_proj) @ V_proj.transpose(0, 1)
            update = update + eta2 * right_ortho

        return update

    def _update_momentum_factor(
        self,
        U: torch.Tensor,
        S: torch.Tensor,
        V: torch.Tensor,
        G: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
        """Fast 2-QR update; falls back to explicit full update when needed.

        Returns
        -------
        U_next, S_next, V_next, used_fallback, qr_failed
        """
        r = int(U.shape[1])
        m, n = int(G.shape[0]), int(G.shape[1])

        if 2 * r > min(m, n):
            U_next, S_next, V_next = self._update_momentum_factor_full(U, S, V, G, beta)
            return U_next, S_next, V_next, True, False

        orig_dtype = G.dtype
        work_dtype = self._work_dtype(orig_dtype)
        U_w = U.to(dtype=work_dtype)
        S_w = S.to(dtype=work_dtype)
        V_w = V.to(dtype=work_dtype)
        G_w = G.to(dtype=work_dtype)

        try:
            GV = G_w.matmul(V_w)
            row_block = torch.cat([U_w, GV], dim=1)
            U_prime, R_U = torch.linalg.qr(row_block, mode="reduced")

            GTU = G_w.transpose(0, 1).matmul(U_w)
            col_block = torch.cat([V_w, GTU], dim=1)
            V_prime, R_V = torch.linalg.qr(col_block, mode="reduced")

            beta_sigma = torch.diag(beta * S_w)
            UTGV = U_w.transpose(0, 1).matmul(G_w).matmul(V_w)
            eye_r = torch.eye(r, device=G.device, dtype=work_dtype)
            zero_r = torch.zeros(r, r, device=G.device, dtype=work_dtype)

            top_left = beta_sigma - UTGV
            top_row = torch.cat([top_left, eye_r], dim=1)
            bottom_row = torch.cat([eye_r, zero_r], dim=1)
            B = torch.cat([top_row, bottom_row], dim=0)

            mid = R_U.matmul(B).matmul(R_V.transpose(0, 1))
            U2, S2, Vh2 = torch.linalg.svd(mid, full_matrices=False)
            U2 = U2[:, :r]
            S2 = S2[:r]
            V2 = Vh2[:r, :].transpose(0, 1).contiguous()

            U_next = U_prime.matmul(U2).to(dtype=orig_dtype)
            S_next = S2.to(dtype=orig_dtype)
            V_next = V_prime.matmul(V2).to(dtype=orig_dtype)
            return U_next, S_next, V_next, False, False
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message or "cuda" in message and "memory" in message:
                raise
            U_next, S_next, V_next = self._update_momentum_factor_full(U, S, V, G, beta)
            return U_next, S_next, V_next, True, True

    @staticmethod
    def _update_momentum_factor_full(
        U: torch.Tensor,
        S: torch.Tensor,
        V: torch.Tensor,
        G: torch.Tensor,
        beta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = int(U.shape[1])
        orig_dtype = G.dtype
        work_dtype = MomentumFactorizedSGD._work_dtype(orig_dtype)
        U_w = U.to(dtype=work_dtype)
        S_w = S.to(dtype=work_dtype)
        V_w = V.to(dtype=work_dtype)
        G_w = G.to(dtype=work_dtype)

        UUTG = U_w @ (U_w.transpose(0, 1) @ G_w)
        GVVT = (G_w @ V_w) @ V_w.transpose(0, 1)
        G_hat = UUTG + GVVT - (UUTG @ V_w) @ V_w.transpose(0, 1)
        M_new = G_hat + beta * ((U_w * S_w.unsqueeze(0)) @ V_w.transpose(0, 1))
        U_new, S_new, V_new = MomentumFactorizedSGD._truncated_svd(M_new.to(dtype=orig_dtype), r)
        return U_new, S_new, V_new

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return lightweight optimizer diagnostics for experiment logs."""
        num_tensors = 0
        initialized = 0
        total_fallbacks = 0
        total_qr_failures = 0
        clipped = 0
        ranks = []

        for group in self.param_groups:
            for p in group["params"]:
                num_tensors += 1
                state = self.state.get(p, {})
                if state:
                    initialized += 1
                    total_fallbacks += int(state.get("fallback_count", 0))
                    total_qr_failures += int(state.get("qr_failure_count", 0))
                    clipped += int(bool(state.get("rank_clipped", False)))
                    if "rank" in state:
                        ranks.append(int(state["rank"]))

        return {
            "num_tensors": num_tensors,
            "initialized_tensors": initialized,
            "rank_clipped_tensors": clipped,
            "total_fallbacks": total_fallbacks,
            "total_qr_failures": total_qr_failures,
            "min_rank": min(ranks) if ranks else None,
            "max_rank": max(ranks) if ranks else None,
        }
