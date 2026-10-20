from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

from .signals import SCHEMA_VERSION, FrameSample

T = TypeVar("T")


class SessionRecorder:
    """Append FrameSamples to a .jsonl file. Use as a context manager."""

    def __init__(self, path: str | Path, session_id: str,
                 meta: Optional[dict[str, Any]] = None, flush_every: int = 30):
        self.path = Path(path)
        self.session_id = session_id
        self.meta = meta or {}
        self.flush_every = flush_every
        self._fh = None
        self._count = 0

    def __enter__(self) -> "SessionRecorder":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        header = {
            "type": "header",
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_at": time.time(),
            "meta": self.meta,
        }
        self._fh.write(json.dumps(header) + "\n")
        return self

    def write(self, sample: FrameSample) -> None:
        if self._fh is None:
            raise RuntimeError("SessionRecorder must be used as a context manager")
        self._fh.write(json.dumps({"type": "sample", **sample.to_dict()}) + "\n")
        self._count += 1
        if self._count % self.flush_every == 0:
            self._fh.flush()

    @property
    def count(self) -> int:
        return self._count

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


class SessionReplayer:
    """Read a recorded session and feed it through a pipeline."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open(encoding="utf-8") as fh:
            first = fh.readline()
        if not first:
            raise ValueError(f"{self.path} is empty")
        header = json.loads(first)
        if header.get("type") != "header":
            raise ValueError(f"{self.path} has no header line")
        if header.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Schema version {header.get('schema_version')} != {SCHEMA_VERSION}; "
                "re-record or add a migration."
            )
        self.header = header

    @property
    def session_id(self) -> str:
        return self.header["session_id"]

    @property
    def meta(self) -> dict[str, Any]:
        return self.header.get("meta", {})

    def samples(self) -> Iterator[FrameSample]:
        with self.path.open(encoding="utf-8") as fh:
            next(fh)  # skip header
            for line_no, line in enumerate(fh, start=2):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    raise ValueError(f"Corrupt JSON at {self.path}:{line_no}") from None
                if rec.pop("type", None) == "sample":
                    yield FrameSample.from_dict(rec)

    def replay(self, step: Callable[[FrameSample], T], realtime: bool = False) -> list[T]:
        """Run `step` on every sample in order and collect its non-None outputs."""
        outputs: list[T] = []
        prev_ts: Optional[float] = None
        for s in self.samples():
            if realtime and prev_ts is not None:
                time.sleep(max(0.0, s.ts - prev_ts))
            prev_ts = s.ts
            out = step(s)
            if out is not None:
                outputs.append(out)
        return outputs