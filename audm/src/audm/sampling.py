"""Duration grouping that preserves each optimizer update's sampled examples."""

import bisect
import itertools
import math

from torch.utils.data import Sampler, Subset


def duration_at(part, index):
    if isinstance(part, Subset):
        return duration_at(part.dataset, part.indices[index])
    values = getattr(part, "durations", None)
    value = values[index] if values is not None else part[index].get("duration")
    return float(value) if value is not None and math.isfinite(value) else 30.0


class UpdateDurationSampler(Sampler):
    """Sort each language inside one accumulation window, never across updates.

    Every microbatch >= the quota window retains exact quotas. Smaller batches
    retain the original quota windows. Tail samples are never dropped. The
    wrapped sampler controls coverage, rank partitioning and deterministic epochs.
    """

    def __init__(self, sampler, parts, microbatch, accumulation):
        self.sampler, self.parts = sampler, parts
        self.window = microbatch * accumulation
        self.block = max(microbatch, sampler.batch_size)
        if self.window % self.block or self.block % sampler.batch_size:
            raise ValueError("Duration grouping requires complete quota windows per update")
        self.ends = list(itertools.accumulate(len(p) for p in parts))
        self.quotas = {
            lang: n * self.block // sampler.batch_size
            for lang, n in sampler.language_batch_counts.items()
        }

    def __len__(self):
        return len(self.sampler)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        source = iter(self.sampler)
        while indices := list(itertools.islice(source, self.window)):
            if len(indices) != self.window:
                yield from indices
                continue
            groups = {lang: [] for lang in self.quotas}
            for index in indices:
                p = bisect.bisect_right(self.ends, index)
                part = self.parts[p]
                local = index - (self.ends[p - 1] if p else 0)
                groups[getattr(part, "language", None) or "und"].append(
                    (duration_at(part, local), index)
                )
            for values in groups.values():
                values.sort()  # deterministic tie-break by dataset index
            for batch in range(self.window // self.block):
                for lang, quota in self.quotas.items():
                    yield from (i for _, i in groups[lang][batch * quota : (batch + 1) * quota])
