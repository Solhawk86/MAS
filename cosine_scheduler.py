import numpy as np


class CosineScheduler:
    """轻量版余弦调度器，摘自 DINOv3 代码，避免依赖 FSDP 模块。"""

    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
        trunc_extra=0.0,
    ):
        super().__init__()
        self.final_value = float(final_value)
        self.total_iters = int(total_iters)

        if self.total_iters <= 0:
            raise ValueError("total_iters must be > 0")

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        if trunc_extra == 0.0:
            remaining = self.total_iters - warmup_iters - freeze_iters
            iters = np.arange(max(remaining, 1))
            schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(len(iters), 1)))
            schedule = schedule[:remaining]
        else:
            cosine_steps = self.total_iters - warmup_iters - freeze_iters
            iters = np.linspace(0, np.pi, int((1 + trunc_extra) * cosine_steps))[:cosine_steps]
            schedule = np.cos(iters)
            schedule = (schedule + 1) / 2
            schedule = (schedule - schedule[-1]) / (1 - schedule[-1])
            schedule = schedule * (base_value - final_value) + final_value

        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule), dtype=np.float64)

        if len(self.schedule) < self.total_iters:
            tail = np.full((self.total_iters - len(self.schedule),), final_value, dtype=np.float64)
            self.schedule = np.concatenate((self.schedule, tail), dtype=np.float64)
        elif len(self.schedule) > self.total_iters:
            self.schedule = self.schedule[: self.total_iters]

    def __getitem__(self, iteration: int) -> float:
        if iteration >= self.total_iters:
            return self.final_value
        return float(self.schedule[iteration])
