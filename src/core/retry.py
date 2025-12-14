from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 10.0
    jitter_frac: float = 0.2

    retry_http_statuses: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_delay_s(*, attempt: int, policy: RetryPolicy, rng: random.Random, retry_after_s: float | None = None) -> float:
    """Compute exponential backoff delay with bounded jitter.

    attempt is 1-indexed (1 == first retry sleep).
    """
    if retry_after_s is not None:
        return _clamp(float(retry_after_s), 0.0, float(policy.max_delay_s))

    base = float(policy.base_delay_s) * (2 ** max(int(attempt) - 1, 0))
    base = _clamp(base, 0.0, float(policy.max_delay_s))

    jitter = float(policy.jitter_frac)
    if jitter <= 0:
        return base

    lo = base * (1.0 - jitter)
    hi = base * (1.0 + jitter)
    return _clamp(rng.uniform(lo, hi), 0.0, float(policy.max_delay_s))


def retry_call(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    is_retryable: Callable[[T], bool] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    retry_after_s: Callable[[T], float | None] | None = None,
    on_retry: Callable[[int, float], None] | None = None,
) -> T:
    """Retry a callable with exponential backoff + jitter.

    - If `is_retryable` is provided, it decides whether to retry based on the return value.
    - Exceptions are not retried by default; callers can wrap exceptions into a retryable return,
      or provide their own `fn` that handles exceptions.
    """
    if rng is None:
        rng = random.Random(0)

    max_attempts = int(policy.max_attempts)
    if max_attempts <= 0:
        return fn()

    attempt = 0
    while True:
        attempt += 1
        value = fn()

        if is_retryable is None or not is_retryable(value):
            return value

        if attempt >= max_attempts:
            return value

        ra = retry_after_s(value) if retry_after_s is not None else None
        delay = compute_delay_s(attempt=attempt, policy=policy, rng=rng, retry_after_s=ra)
        if on_retry is not None:
            on_retry(attempt, delay)
        sleep_fn(delay)


def parse_retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        return None
