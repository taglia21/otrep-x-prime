import random

from core.retry import RetryPolicy, compute_delay_s, retry_call


def test_retry_call_respects_max_attempts():
    attempts = 0
    sleeps: list[float] = []

    def fn() -> int:
        nonlocal attempts
        attempts += 1
        return attempts

    policy = RetryPolicy(max_attempts=3, base_delay_s=0.1, max_delay_s=1.0, jitter_frac=0.0)

    out = retry_call(
        fn,
        policy=policy,
        is_retryable=lambda v: v < 3,
        sleep_fn=lambda s: sleeps.append(float(s)),
        rng=random.Random(0),
    )

    assert out == 3
    assert attempts == 3
    assert sleeps == [0.1, 0.2]


def test_compute_delay_is_capped():
    policy = RetryPolicy(max_attempts=10, base_delay_s=10.0, max_delay_s=1.0, jitter_frac=0.2)
    rng = random.Random(0)

    d = compute_delay_s(attempt=5, policy=policy, rng=rng)
    assert 0.0 <= d <= 1.0
