import torch
from torch.optim import Optimizer


class MomentumFactorizedSGD(Optimizer):
    r"""
    Momentum Factorized SGD (MoFaSGD) for 2D matrix parameters.

    State per parameter is stored as tensors U, S, V such that
        M ~= U diag(S) V^T.

    The default update uses U V^T by setting use_ones_for_nonzero_s=True,
    which matches the spectrally normalized update form used in the paper.
    Set use_ones_for_nonzero_s=False to use the reciprocal-S variant from the
    public reference implementation.
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        rank=16,
        beta=0.9,
        eta1=1.0,
        eta2=0.0,
        use_current_projection=False,
        use_ones_for_nonzero_s=True,
        eps=1e-4,
        nesterov=False,
        max_value=10000.0,
        weight_decay=0.0,
        init_from_grad=True,
    ):
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        defaults = dict(
            lr=lr,
            rank=rank,
            beta=beta,
            eta1=eta1,
            eta2=eta2,
            use_current_projection=use_current_projection,
            use_ones_for_nonzero_s=use_ones_for_nonzero_s,
            eps=eps,
            nesterov=nesterov,
            max_value=max_value,
            weight_decay=weight_decay,
            init_from_grad=init_from_grad,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _effective_rank(shape, requested_rank):
        m, n = shape
        # The efficient QR update uses 2r columns on both sides.  For skinny
        # matrices such as NeRF's first pts_linear, clip rank so 2r <= min(m,n).
        max_rank = max(1, min(m, n) // 2)
        return max(1, min(int(requested_rank), max_rank))

    @staticmethod
    def _truncated_svd(x, rank):
        # SVD is computed in fp32 for numerical stability if needed.  Returned
        # tensors are cast back to the parameter dtype.
        orig_dtype = x.dtype
        x_svd = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        U, S, Vh = torch.linalg.svd(x_svd, full_matrices=False)
        U = U[:, :rank].to(dtype=orig_dtype)
        S = S[:rank].to(dtype=orig_dtype)
        V = Vh[:rank, :].transpose(0, 1).contiguous().to(dtype=orig_dtype)
        return U, S, V

    def _init_state(self, p, group):
        if p.dim() != 2:
            raise RuntimeError("MomentumFactorizedSGD only supports 2D parameters.")
        rank = self._effective_rank(p.shape, group["rank"])
        source = p.grad if (group["init_from_grad"] and p.grad is not None) else p.data
        U, S, V = self._truncated_svd(source.detach(), rank)
        state = self.state[p]
        state["U"] = U.clone()
        state["S"] = S.clone()
        state["V"] = V.clone()
        state["rank"] = rank
        state["step"] = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eta1 = group["eta1"]
            eta2 = group["eta2"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise RuntimeError("MomentumFactorizedSGD only supports 2D parameters.")

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                    state = self.state[p]

                G = p.grad.detach()
                U, S, V = state["U"], state["S"], state["V"]

                U_prev, V_prev = U, V
                U_next, S_next, V_next = self._update_momentum_factor(U, S, V, G, beta)

                if group["nesterov"]:
                    U_next, S_next, V_next = self._update_momentum_factor(
                        U_next, S_next, V_next, G, beta
                    )

                state["U"] = U_next
                state["S"] = S_next
                state["V"] = V_next
                state["rank"] = U_next.shape[1]
                state["step"] += 1

                if weight_decay != 0:
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
                    use_current_projection=group["use_current_projection"],
                    use_ones_for_nonzero_s=group["use_ones_for_nonzero_s"],
                    eps=group["eps"],
                    max_value=group["max_value"],
                )
                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _make_update(
        G,
        U_prev,
        V_prev,
        U_next,
        S_next,
        V_next,
        eta1,
        eta2,
        use_current_projection,
        use_ones_for_nonzero_s,
        eps,
        max_value,
    ):
        non_zero = S_next.abs() > eps
        scale = torch.zeros_like(S_next)
        if use_ones_for_nonzero_s:
            scale[non_zero] = 1.0
        else:
            scale[non_zero] = 1.0 / S_next[non_zero]
            scale = torch.clamp(scale, max=max_value)

        low_rank_update = (U_next * scale.unsqueeze(0)) @ V_next.transpose(0, 1)
        update = eta1 * low_rank_update

        if eta2 != 0.0:
            if use_current_projection:
                U_proj, V_proj = U_next, V_next
            else:
                U_proj, V_proj = U_prev, V_prev
            left_ortho = G - U_proj @ (U_proj.transpose(0, 1) @ G)
            right_ortho = left_ortho - (left_ortho @ V_proj) @ V_proj.transpose(0, 1)
            update = update + eta2 * right_ortho

        return update

    def _update_momentum_factor(self, U, S, V, G, beta):
        r = U.shape[1]
        m, n = G.shape
        if 2 * r > min(m, n):
            return self._update_momentum_factor_full(U, S, V, G, beta)

        try:
            GV = G.matmul(V)
            row_block = torch.cat([U, GV], dim=1)
            U_prime, R_U = torch.linalg.qr(row_block, mode="reduced")

            GTU = G.transpose(0, 1).matmul(U)
            col_block = torch.cat([V, GTU], dim=1)
            V_prime, R_V = torch.linalg.qr(col_block, mode="reduced")

            beta_sigma = torch.diag(beta * S)
            UTGV = U.transpose(0, 1).matmul(G).matmul(V)
            eye_r = torch.eye(r, device=G.device, dtype=G.dtype)
            zero_r = torch.zeros(r, r, device=G.device, dtype=G.dtype)

            top_left = beta_sigma - UTGV
            top_row = torch.cat([top_left, eye_r], dim=1)
            bottom_row = torch.cat([eye_r, zero_r], dim=1)
            B = torch.cat([top_row, bottom_row], dim=0)

            mid = R_U.matmul(B).matmul(R_V.transpose(0, 1))
            U2, S2, Vh2 = torch.linalg.svd(mid, full_matrices=False)
            U2 = U2[:, :r]
            S2 = S2[:r]
            V2 = Vh2[:r, :].transpose(0, 1).contiguous()

            U_next = U_prime.matmul(U2)
            V_next = V_prime.matmul(V2)
            return U_next, S2, V_next
        except RuntimeError:
            return self._update_momentum_factor_full(U, S, V, G, beta)

    @staticmethod
    def _update_momentum_factor_full(U, S, V, G, beta):
        r = U.shape[1]
        # Tangent-space projection of the gradient followed by rank-r SVD.
        UUTG = U @ (U.transpose(0, 1) @ G)
        GVVT = (G @ V) @ V.transpose(0, 1)
        G_hat = UUTG + GVVT - (UUTG @ V) @ V.transpose(0, 1)
        M_new = G_hat + beta * ((U * S.unsqueeze(0)) @ V.transpose(0, 1))
        U_new, S_new, V_new = MomentumFactorizedSGD._truncated_svd(M_new, r)
        return U_new, S_new, V_new
