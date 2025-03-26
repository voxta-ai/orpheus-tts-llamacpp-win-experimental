import time
import logging

class Stopwatch:
    def __init__(self, name: str):
        self._name = name
        self._append: list[str] = []

    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        logging.info(f"{self._name} ({time.monotonic() - self.start:.4f}s){'' if len(self._append) == 0 else ' ' + ', '.join(self._append) + ')'}")

    def append(self, text):
        self._append.append(text)