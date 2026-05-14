"""
Rank-aware WSD scheduler with optimizer-rank tracking.

This scheduler is designed for two different cases in one implementation:

1) Progressive-rank optimizer, e.g. auto-cos-inc:
   - The optimizer owns the rank schedule.
   - The scheduler reads optimizer.param_groups[*] metadata:
       current_rank, current_target_rank, rank_saturated, rank_wsd_role.
   - LR stays at peak while rank is still growing.
   - Linear decay starts automatically when the rank becomes saturated.

2) Full-rank Muon / fixed-rank optimizer:
   - The optimizer marks itself as already saturated.
   - The scheduler does NOT grant the late-decay privilege.
   - It starts earlier decay and can optionally finish decay before the final step.

The scheduler itself never changes rank. It only changes LR.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


class RankAwareWarmupStableLinearScheduler:
    """
    Rank-aware WSD LR scheduler.

    LR shape:
        optional micro warmup -> stable plateau -> linear decay -> optional min-LR hold

    The important distinction is `decay_start_step` and `decay_end_step`:

      progressive rank:
        decay_start_step = first step where current_rank >= current_target_rank
        decay_end_step   = total_iters - 1

      full-rank Muon / fixed-rank:
        decay_start_step = fullrank_decay_start_frac * total_iters
        decay_end_step   = fullrank_decay_end_frac   * total_iters

    If fullrank_decay_end_frac < 1.0, Muon reaches min_lr_ratio early and then
    stays there. This is the actual "fast decay" version, not just early onset.
    """

    def __init__(
        self,
        base_lr_adam: float,
        base_lr_muon: float,
        total_iters: int,
        base_N_rand: int,
        warmup_steps: int = 0,
        decay_start_step: Optional[int] = None,
        decay_end_step: Optional[int] = None,
        min_lr_ratio: float = 0.1,
        default_decay_start_frac: float = 0.8,
        default_decay_end_frac: float = 1.0,
        fullrank_decay_start_frac: float = 0.2,
        fullrank_decay_end_frac: float = 0.6,
        rank_saturation_patience_steps: int = 0,
        require_all_progressive_groups: bool = True,
    ) -> None:
        if total_iters <= 0:
            raise ValueError(f"total_iters must be > 0, got {total_iters}")
        if base_N_rand <= 0:
            raise ValueError(f"base_N_rand must be > 0, got {base_N_rand}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

        for name, value in {
            "default_decay_start_frac": default_decay_start_frac,
            "default_decay_end_frac": default_decay_end_frac,
            "fullrank_decay_start_frac": fullrank_decay_start_frac,
            "fullrank_decay_end_frac": fullrank_decay_end_frac,
        }.items():
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        self.base_lr_adam = float(base_lr_adam)
        self.base_lr_muon = float(base_lr_muon)
        self.total_iters = int(total_iters)
        self.base_N_rand = int(base_N_rand)
        self.warmup_steps = min(int(warmup_steps), max(0, self.total_iters - 1))
        self.min_lr_ratio = float(min_lr_ratio)

        self.default_decay_start_frac = float(default_decay_start_frac)
        self.default_decay_end_frac = float(default_decay_end_frac)
        self.fullrank_decay_start_frac = float(fullrank_decay_start_frac)
        self.fullrank_decay_end_frac = float(fullrank_decay_end_frac)
        self.rank_saturation_patience_steps = max(0, int(rank_saturation_patience_steps))
        self.require_all_progressive_groups = bool(require_all_progressive_groups)

        self._explicit_decay_start = decay_start_step is not None and int(decay_start_step) >= 0
        self._explicit_decay_end = decay_end_step is not None and int(decay_end_step) >= 0

        self.decay_start_step: Optional[int] = None
        self.decay_end_step: Optional[int] = None
        self.decay_start_source = "unresolved"
        self.decay_end_source = "unresolved"

        if self._explicit_decay_start:
            self.decay_start_step = self._clamp_step(int(decay_start_step), lower=self.warmup_steps)
            self.decay_start_source = "explicit"
        if self._explicit_decay_end:
            lower = self.decay_start_step if self.decay_start_step is not None else self.warmup_steps
            self.decay_end_step = self._clamp_step(int(decay_end_step), lower=lower)
            self.decay_end_source = "explicit"

        self._last_rank_status: Dict[str, Any] = {}

    @property
    def effective_total_iters(self) -> int:
        return self.total_iters

    def _clamp_step(self, value: int, lower: int = 0) -> int:
        value = int(value)
        value = max(int(lower), value)
        return min(max(0, self.total_iters - 1), value)

    def _frac_to_step(self, frac: float, lower: int = 0) -> int:
        return self._clamp_step(int(round(float(frac) * self.total_iters)), lower=lower)

    def _set_decay_window(self, start: int, end: int, start_source: str, end_source: str) -> None:
        start = self._clamp_step(start, lower=self.warmup_steps)
        end = self._clamp_step(end, lower=start)
        self.decay_start_step = start
        self.decay_end_step = end
        self.decay_start_source = start_source
        self.decay_end_source = end_source

    @staticmethod
    def _is_tracked_group(group: Dict[str, Any]) -> bool:
        role = group.get("rank_wsd_role")
        return bool(group.get("use_muon", False)) or role in {
            "progressive_lowrank",
            "full_rank_muon",
            "fixed_lowrank",
        }

    @staticmethod
    def _rank_saturated_from_group(group: Dict[str, Any]) -> bool:
        if "rank_saturated" in group:
            return bool(group.get("rank_saturated"))
        current = _safe_int(group.get("current_rank"), default=None)
        target = _safe_int(group.get("current_target_rank"), default=None)
        if current is None or target is None:
            return False
        return current >= target

    def _collect_rank_status(self, optimizer: Optional[Any]) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "has_optimizer": optimizer is not None,
            "has_tracked_group": False,
            "has_progressive_rank": False,
            "has_fullrank_muon": False,
            "has_fixed_lowrank": False,
            "progressive_all_saturated": False,
            "progressive_any_saturated": False,
            "rank_saturated": False,
            "rank_role": None,
            "rank_method": None,
            "current_rank": None,
            "target_rank": None,
            "rank_progress": None,
            "groups": [],
        }

        if optimizer is None or not hasattr(optimizer, "param_groups"):
            return status

        progressive_saturated: List[bool] = []
        ranks: List[int] = []
        targets: List[int] = []
        progresses: List[float] = []

        for group_idx, group in enumerate(optimizer.param_groups):
            if not isinstance(group, dict) or not self._is_tracked_group(group):
                continue

            role = str(group.get("rank_wsd_role", "unknown"))
            is_progressive = bool(group.get("is_progressive_rank", False)) or role == "progressive_lowrank"
            is_fullrank = bool(group.get("is_full_rank", False)) or role == "full_rank_muon"
            is_fixed = role == "fixed_lowrank"
            current = _safe_int(group.get("current_rank"), default=None)
            target = _safe_int(group.get("current_target_rank"), default=None)
            saturated = self._rank_saturated_from_group(group)

            start_rank = _safe_int(group.get("rank_start"), default=None)
            rank_end = _safe_int(group.get("rank_end"), default=target)
            progress = _safe_float(group.get("rank_progress"), default=None)
            if progress is None and current is not None and start_rank is not None and rank_end is not None:
                denom = max(1, int(rank_end) - int(start_rank))
                progress = max(0.0, min(1.0, (int(current) - int(start_rank)) / float(denom)))
            if progress is not None:
                progresses.append(progress)

            if current is not None:
                ranks.append(current)
            if target is not None:
                targets.append(target)
            if is_progressive:
                progressive_saturated.append(saturated)

            status["has_tracked_group"] = True
            status["has_progressive_rank"] = bool(status["has_progressive_rank"] or is_progressive)
            status["has_fullrank_muon"] = bool(status["has_fullrank_muon"] or is_fullrank)
            status["has_fixed_lowrank"] = bool(status["has_fixed_lowrank"] or is_fixed)

            status["groups"].append({
                "group_idx": group_idx,
                "role": role,
                "is_progressive": is_progressive,
                "is_fullrank": is_fullrank,
                "is_fixed_lowrank": is_fixed,
                "current_rank": current,
                "target_rank": target,
                "rank_start": start_rank,
                "rank_end": rank_end,
                "rank_progress": progress,
                "warmup_steps": _safe_int(group.get("warmup_steps"), default=None),
                "saturated": saturated,
                "method": group.get("current_method", group.get("rank_schedule_type")),
                "saturation_step": group.get("rank_saturation_step"),
            })

        if progressive_saturated:
            status["progressive_all_saturated"] = all(progressive_saturated)
            status["progressive_any_saturated"] = any(progressive_saturated)
            status["rank_saturated"] = (
                status["progressive_all_saturated"]
                if self.require_all_progressive_groups
                else status["progressive_any_saturated"]
            )
            status["rank_role"] = "progressive_lowrank"
        elif status["has_fullrank_muon"]:
            status["rank_saturated"] = True
            status["rank_role"] = "full_rank_muon"
        elif status["has_fixed_lowrank"]:
            status["rank_saturated"] = True
            status["rank_role"] = "fixed_lowrank"
        elif status["has_tracked_group"]:
            status["rank_role"] = "unknown_tracked"

        if ranks:
            status["current_rank"] = max(ranks)
        if targets:
            status["target_rank"] = max(targets)
        if progresses:
            # Conservative summary: use the minimum progress across progressive groups.
            status["rank_progress"] = min(progresses)
        if status["groups"]:
            status["rank_method"] = status["groups"][0].get("method")

        return status

    def _maybe_resolve_decay_window(self, step: int, optimizer: Optional[Any]) -> Dict[str, Any]:
        status = self._collect_rank_status(optimizer)
        self._last_rank_status = status

        if self.decay_start_step is not None and self.decay_end_step is not None:
            return status

        # Explicit start but implicit end: use total_iters - 1 unless user supplied decay_end_step.
        if self.decay_start_step is not None and self.decay_end_step is None:
            self.decay_end_step = self._clamp_step(self.total_iters - 1, lower=self.decay_start_step)
            self.decay_end_source = "implicit_final_step"
            return status

        # Progressive rank: wait until actual saturation, then decay to the final step.
        if status["has_progressive_rank"]:
            if status["rank_saturated"]:
                start = int(step) + self.rank_saturation_patience_steps
                end = self.total_iters - 1
                self._set_decay_window(
                    start=start,
                    end=end,
                    start_source="progressive_rank_saturation",
                    end_source="progressive_final_step",
                )
            return status

        # Full-rank Muon: already saturated, so no late-decay privilege.
        if status["has_fullrank_muon"]:
            start = self._frac_to_step(self.fullrank_decay_start_frac, lower=self.warmup_steps)
            end = self._frac_to_step(self.fullrank_decay_end_frac, lower=start)
            self._set_decay_window(
                start=start,
                end=end,
                start_source="full_rank_muon_early_decay",
                end_source="full_rank_muon_fast_decay_end",
            )
            return status

        # Fixed low-rank optimizer: also saturated from the beginning.
        if status["has_fixed_lowrank"]:
            start = self._frac_to_step(self.fullrank_decay_start_frac, lower=self.warmup_steps)
            end = self._frac_to_step(self.fullrank_decay_end_frac, lower=start)
            self._set_decay_window(
                start=start,
                end=end,
                start_source="fixed_rank_early_decay",
                end_source="fixed_rank_fast_decay_end",
            )
            return status

        # Adam / unknown: stable old fallback.
        start = self._frac_to_step(self.default_decay_start_frac, lower=self.warmup_steps)
        end = self._frac_to_step(self.default_decay_end_frac, lower=start)
        self._set_decay_window(
            start=start,
            end=end,
            start_source="fallback_default_frac",
            end_source="fallback_default_end_frac",
        )
        return status

    def _lr_ratio(self, step: int) -> float:
        step = int(max(0, step))

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(max(1, self.warmup_steps))

        if self.decay_start_step is None or self.decay_end_step is None:
            return 1.0

        if step < self.decay_start_step:
            return 1.0

        if step >= self.decay_end_step:
            return self.min_lr_ratio

        decay_span = max(1, int(self.decay_end_step) - int(self.decay_start_step))
        decay_step = max(0, int(step) - int(self.decay_start_step))
        progress = float(decay_step) / float(decay_span)
        return 1.0 - (1.0 - self.min_lr_ratio) * progress

    def step(self, global_step: int, optimizer: Optional[Any] = None) -> Dict[str, Any]:
        step = int(max(0, global_step))
        rank_status = self._maybe_resolve_decay_window(step, optimizer)
        ratio = self._lr_ratio(step)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            phase = -1
            phase_name = "warmup"
        elif self.decay_start_step is None or self.decay_end_step is None:
            phase = 0
            phase_name = "stable_wait_rank" if rank_status.get("has_progressive_rank") else "stable_unresolved"
        elif step < self.decay_start_step:
            phase = 0
            phase_name = "stable_wait_rank" if rank_status.get("has_progressive_rank") else "stable"
        elif step >= self.decay_end_step:
            phase = 2
            phase_name = "min_lr_hold"
        else:
            phase = 1
            phase_name = "linear_decay"

        decay_progress = None
        if self.decay_start_step is not None and self.decay_end_step is not None:
            span = max(1, int(self.decay_end_step) - int(self.decay_start_step))
            decay_progress = max(0.0, min(1.0, (step - int(self.decay_start_step)) / float(span)))

        return {
            "lr_adam": self.base_lr_adam * ratio,
            "lr_muon": self.base_lr_muon * ratio,
            "N_rand": self.base_N_rand,
            "phase": phase,
            "phase_name": phase_name,
            "lr_ratio": ratio,
            "decay_start_step": self.decay_start_step,
            "decay_end_step": self.decay_end_step,
            "decay_start_source": self.decay_start_source,
            "decay_end_source": self.decay_end_source,
            "decay_progress": decay_progress,
            "rank_current": rank_status.get("current_rank"),
            "rank_target": rank_status.get("target_rank"),
            "rank_progress": rank_status.get("rank_progress"),
            "rank_saturated": rank_status.get("rank_saturated"),
            "rank_role": rank_status.get("rank_role"),
            "rank_method": rank_status.get("rank_method"),
            "rank_groups": rank_status.get("groups", []),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay_start_step": self.decay_start_step,
            "decay_end_step": self.decay_end_step,
            "decay_start_source": self.decay_start_source,
            "decay_end_source": self.decay_end_source,
            "last_rank_status": self._last_rank_status,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict:
            return
        self.decay_start_step = state_dict.get("decay_start_step", self.decay_start_step)
        self.decay_end_step = state_dict.get("decay_end_step", self.decay_end_step)
        self.decay_start_source = state_dict.get("decay_start_source", self.decay_start_source)
        self.decay_end_source = state_dict.get("decay_end_source", self.decay_end_source)
        self._last_rank_status = state_dict.get("last_rank_status", {})

    def describe(self) -> str:
        start = "auto" if self.decay_start_step is None else str(self.decay_start_step)
        end = "auto" if self.decay_end_step is None else str(self.decay_end_step)
        return (
            "[RankWSD-AutoV2] "
            f"total_iters={self.total_iters} warmup_steps={self.warmup_steps} "
            f"decay_start_step={start} source={self.decay_start_source} "
            f"decay_end_step={end} source={self.decay_end_source} "
            f"min_lr_ratio={self.min_lr_ratio:.6g} base_N_rand={self.base_N_rand} "
            f"fullrank_decay_start_frac={self.fullrank_decay_start_frac:.4g} "
            f"fullrank_decay_end_frac={self.fullrank_decay_end_frac:.4g} "
            f"default_decay_start_frac={self.default_decay_start_frac:.4g} "
            f"default_decay_end_frac={self.default_decay_end_frac:.4g} "
            f"rank_saturation_patience_steps={self.rank_saturation_patience_steps}"
        )
