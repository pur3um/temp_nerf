import math
from typing import Dict
# micro-warmup + cosine decay

class WarmupCosineScheduler:
    """
    Strong baseline for INR runs:
        micro warmup -> cosine decay

    The scheduler is stateless and returns both Muon-branch and Adam-branch
    learning rates using the same ratio but different base LRs.
    """

    def __init__(
        self,
        base_lr_adam: float,
        base_lr_muon: float,
        total_iters: int,
        base_N_rand: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if total_iters <= 0:
            raise ValueError(f"total_iters must be > 0, got {total_iters}")
        if base_N_rand <= 0:
            raise ValueError(f"base_N_rand must be > 0, got {base_N_rand}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

        self.base_lr_adam = float(base_lr_adam)
        self.base_lr_muon = float(base_lr_muon)
        self.total_iters = int(total_iters)
        self.base_N_rand = int(base_N_rand)
        self.warmup_steps = min(int(warmup_steps), max(0, self.total_iters - 1))
        self.min_lr_ratio = float(min_lr_ratio)
        self._cosine_span = max(1, (self.total_iters - 1) - self.warmup_steps)

    @property
    def effective_total_iters(self) -> int:
        return self.total_iters

    def _lr_ratio(self, step: int) -> float:
        step = int(max(0, step))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(max(1, self.warmup_steps))

        cosine_step = min(max(step - self.warmup_steps, 0), self._cosine_span)
        progress = float(cosine_step) / float(self._cosine_span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def step(self, global_step: int) -> Dict[str, float]:
        step = int(max(0, global_step))
        ratio = self._lr_ratio(step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            phase = -1
            phase_name = "warmup"
        else:
            phase = 0
            phase_name = "cosine"

        return {
            "lr_adam": self.base_lr_adam * ratio,
            "lr_muon": self.base_lr_muon * ratio,
            "N_rand": self.base_N_rand,
            "phase": phase,
            "phase_name": phase_name,
            "lr_ratio": ratio,
        }

    def describe(self) -> str:
        return (
            "[WarmupCosine] "
            f"total_iters={self.total_iters} warmup_steps={self.warmup_steps} "
            f"min_lr_ratio={self.min_lr_ratio:.6g} base_N_rand={self.base_N_rand}"
        )
