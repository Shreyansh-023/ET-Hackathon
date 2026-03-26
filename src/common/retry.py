from __future__ import annotations

from collections.abc import Callable
import time
from typing import TypeVar

from src.common.errors import PipelineError

T = TypeVar("T")


def run_with_retry(
    fn: Callable[[], T],
    *,
    retries: int,
    delay_seconds: float = 0.4,
    backoff: float = 2.0,
    retry_on: tuple[type[Exception], ...] = (PipelineError,),
) -> T:
    attempt = 0
    while True:
        try:
            return fn()
        except retry_on:
            attempt += 1
            if attempt > retries:
                raise
            sleep_for = delay_seconds * (backoff ** (attempt - 1))
            time.sleep(sleep_for)
