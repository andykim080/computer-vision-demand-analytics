"""Directional finite-line counting, independent of detector/runtime dependencies."""
from dataclasses import dataclass
from math import hypot, ceil


def confirmation_frames(config, fps):
    """Honor elapsed-time confirmation when supplied; preserve legacy frame configs."""
    if 'confirm_seconds' in config:
        return max(1, ceil(config['confirm_seconds'] * fps - 1e-9))
    return config.get('confirm_frames', 3)


@dataclass
class State:
    side: int
    point: tuple
    last_frame: int
    candidate_side: int = 0
    candidate_frames: int = 0
    crossing_pending: bool = False


class LineCounter:
    """Count confirmed side changes across a finite segment using bottom-center points.

    A deadband and consecutive confirmation suppress jitter. Observations separated
    by a long gap are reset rather than inferring unseen crossings. An optional
    shorter initial confirmation supports objects already approaching a line at
    clip start; destination-side confirmation is still required.
    """

    def __init__(self, start, end, entry_side="positive", deadband=8,
                 confirm_frames=3, max_gap_frames=15, initial_confirm_frames=None):
        self.start, self.end = tuple(start), tuple(end)
        self.dx = end[0] - start[0]
        self.dy = end[1] - start[1]
        self.length = hypot(self.dx, self.dy)
        if self.length == 0:
            raise ValueError("Counting line must have distinct endpoints")
        if entry_side not in ("positive", "negative"):
            raise ValueError("entry_side must be positive or negative")
        if deadband < 0 or confirm_frames < 1 or max_gap_frames < 1:
            raise ValueError("Invalid counting thresholds")
        self.entry_side = 1 if entry_side == "positive" else -1
        self.deadband, self.confirm_frames = deadband, confirm_frames
        self.initial_confirm_frames = confirm_frames if initial_confirm_frames is None else initial_confirm_frames
        if self.initial_confirm_frames < 1:
            raise ValueError("Initial confirmation frames must be positive")
        self.max_gap_frames = max_gap_frames
        self.states = {}

    def distance(self, point):
        return (self.dx * (point[1] - self.start[1]) -
                self.dy * (point[0] - self.start[0])) / self.length

    def crosses_segment(self, previous, current):
        d0, d1 = self.distance(previous), self.distance(current)
        if d0 * d1 > 0 or d0 == d1:
            return False
        t = d0 / (d0 - d1)
        x = previous[0] + t * (current[0] - previous[0])
        y = previous[1] + t * (current[1] - previous[1])
        projection = ((x - self.start[0]) * self.dx +
                      (y - self.start[1]) * self.dy) / self.length**2
        return 0 <= projection <= 1

    def prune(self, frame):
        self.states = {k: v for k, v in self.states.items()
                       if frame - v.last_frame <= self.max_gap_frames}

    def update(self, track_id, point, frame):
        distance = self.distance(point)
        side = 1 if distance > self.deadband else -1 if distance < -self.deadband else 0
        state = self.states.get(track_id)
        if state is None or frame - state.last_frame > self.max_gap_frames:
            self.states[track_id] = State(0, point, frame)
            state = self.states[track_id]
        if side == 0:
            state.candidate_side = 0
            state.candidate_frames = 0
            state.last_frame = frame
            return None
        # Keep the previous non-deadband point while traversing the deadband.
        crossing = self.crosses_segment(state.point, point)
        # A valid crossing stays pending through deadband jitter or a short
        # detection gap until its destination side is confirmed. Returning to
        # the original committed side cancels it. Overwriting this latch when
        # candidate confirmation restarts silently loses real crossings.
        if state.side and side == state.side:
            state.crossing_pending = False
        elif state.side and crossing:
            state.crossing_pending = True
        if side != state.candidate_side or frame - state.last_frame > 1:
            state.candidate_side, state.candidate_frames = side, 1
        else:
            state.candidate_frames += 1
        state.last_frame = frame
        state.point = point
        required = self.initial_confirm_frames if state.side == 0 else self.confirm_frames
        if state.candidate_frames < required:
            return None
        old_side = state.side
        state.side = side
        if not old_side:
            state.crossing_pending = False
        if old_side and old_side != side and state.crossing_pending:
            state.crossing_pending = False
            return "entry" if side == self.entry_side else "exit"
        return None
