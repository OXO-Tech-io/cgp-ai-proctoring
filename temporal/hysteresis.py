from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class State(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    DEVIATING = "DEVIATING"
    RETURNING = "RETURNING"

@dataclass(frozen=True)
class HysteresisConfig:
    enter: float = 1.0
    exit : float = 0.5
    suspect_s: float = 0.3
    return_s: float = 0.5
    bridge_s: float = 1.0
    lose_s: float = 5.0
    split_after_gap: bool = True

    def __post_init(self) -> None:
        if not self.exit < self.enter:
            raise ValueError("exit must be lower than enrter")
        if min(self.suspect_s, self.return_s, self.bridge_s) < 0 or self.lose_s <= self.bridge_s:
            raise ValueError("invalid dwell/gap times")


@dataclass(frozen=True)
class Transition:
    ts: float
    old: State
    new: State
    reason: str


@dataclass(frozen=True)
class Episode:
    start_ts: float                  
    confirmed_ts: Optional[float]    
    end_ts: float
    deviating_s: float               
    peak: float                      
    end_reason: str                  


class HysteresisMachine:
    def __init__(self, cfg: HysteresisConfig = HysteresisConfig()):
        self.cfg = cfg
        self.state = State.NORMAL
        self._reset_episode()
        self._last_known_ts: Optional[float] = None
        self._resume = False

    def _reset_episode(self) -> None:
        self._start: Optional[float] = None
        self._confirmed: Optional[float] = None
        self._dev_since: Optional[float] = None
        self._dev_total = 0.0
        self._peak = 0.0
        self._since = 0.0  

    def _go(self, ts: float, new: State, reason: str, out: list) -> None:
        out.append(Transition(ts, self.state, new, reason))
        self.state = new

    def _end(self, ts: float, reason: str) -> Episode:
        if self._dev_since is not None:
            self._dev_total += ts - self._dev_since
        ep = Episode(self._start, self._confirmed, ts, self._dev_total, self._peak, reason)
        self._reset_episode()
        return ep

    def abort(self, ts: float, reason: str):
        """Force-close any open episode (e.g. another detector took over).
        Returns (transitions, episode_or_None)."""
        out: list[Transition] = []
        if self.state == State.NORMAL:
            return out, None
        episode = self._end(ts, reason)
        self._go(ts, State.NORMAL, reason, out)
        self._resume = False
        return out, episode

    def update(self, ts: float, score: Optional[float]):
        """Returns (transitions, episode_or_None)."""
        c = self.cfg
        out: list[Transition] = []
        episode: Optional[Episode] = None

        if score is None:
            if self._last_known_ts is None:
                return out, None
            gap = ts - self._last_known_ts
            if gap > c.lose_s:
                if self.state != State.NORMAL:
                    episode = self._end(ts, "signal_lost")
                    self._go(ts, State.NORMAL, "signal_lost", out)
                return out, episode
            if gap <= c.bridge_s:
                return out, None                   
            if self.state == State.SUSPECT:        # cannot confirm through a gap
                episode = self._end(ts, "dropped")
                self._go(ts, State.NORMAL, "gap", out)
            elif self.state == State.RETURNING and c.split_after_gap:
                # the look had already ended; whatever follows the blind gap is a new one
                episode = self._end(self._last_known_ts, "gap")
                self._go(ts, State.NORMAL, "gap", out)
                self._resume = False
                return out, episode
            self._resume = True                    # DEVIATING/RETURNING: freeze
            return out, episode
        else:
            self._last_known_ts = ts

        if self._resume:                           
            self._since = ts
            self._resume = False

        if self.state != State.NORMAL:
            self._peak = max(self._peak, score)

        if self.state == State.NORMAL:
            if score >= c.enter:
                self._reset_episode()
                self._start, self._since, self._peak = ts, ts, score
                self._go(ts, State.SUSPECT, "crossed_enter", out)

        elif self.state == State.SUSPECT:
            if score < c.exit:
                episode = self._end(ts, "dropped")
                self._go(ts, State.NORMAL, "too_brief", out)
            elif ts - self._since >= c.suspect_s:
                self._confirmed = self._dev_since = ts
                self._go(ts, State.DEVIATING, "sustained", out)

        elif self.state == State.DEVIATING:
            if score < c.exit:
                self._dev_total += ts - self._dev_since
                self._dev_since = None
                self._since = ts
                self._go(ts, State.RETURNING, "below_exit", out)

        elif self.state == State.RETURNING:
            if score >= c.enter:
                self._dev_since = ts
                self._go(ts, State.DEVIATING, "resumed", out)
            elif score >= c.exit:
                self._since = ts                   
            elif ts - self._since >= c.return_s:
                episode = self._end(ts, "returned")
                self._go(ts, State.NORMAL, "returned", out)

        return out, episode