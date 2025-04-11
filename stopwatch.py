import time
import logging

logger = logging.getLogger(__name__)

class Stopwatch:
    def __init__(self, name: str, auto_start: bool = True):
        self._enabled = logger.isEnabledFor(logging.INFO)
        self.auto_start = auto_start
        self._name = name
        self._append: list[str] = []
        self.timer_start = None
        self.timer_sum = 0

    def __enter__(self):
        if not self._enabled:
            return self
        if self.auto_start:
            self.timer_start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self._enabled:
            return self
        if exc_type is not None:
            return
        if self.timer_start is not None:
            delay = time.perf_counter_ns() - self.timer_start
            self.timer_sum += delay
            self.timer_start = None
        timer_ms = self.timer_sum * 1e-6
        logger.debug(f"{self._name} ({timer_ms:.4f}ms){'' if len(self._append) == 0 else ' ' + ', '.join(self._append) + ')'}")
    
    def start(self):
        if not self._enabled:
            return
        self.timer_start = time.perf_counter_ns()
    
    def stop(self):
        if not self._enabled:
            return
        if self.timer_start is not None:
            delay = time.perf_counter_ns() - self.timer_start
            self.timer_sum += delay
            self.timer_start = None
    
    def duration(self):
        if not self._enabled:
            return 0
        if self.timer_start is not None:
            delay = time.perf_counter_ns() - self.timer_start
            return self.timer_sum + delay
        return self.timer_sum

    def append(self, text):
        if not self._enabled:
            return
        self._append.append(text)
