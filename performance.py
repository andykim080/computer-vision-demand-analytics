"""Arrival-paced playback measurements; no frame dropping or detection caching."""
import csv
import math
import time


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


class PlaybackTiming:
    def __init__(self, fps, paced=False, clock=time.perf_counter, sleep=time.sleep):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Playback FPS must be positive and finite')
        self.fps, self.paced, self.clock, self.sleep = fps, paced, clock, sleep
        self.epoch = clock()
        self.rows = []

    def begin_frame(self, frame):
        if self.paced:
            wait = self.epoch + frame / self.fps - self.clock()
            if wait > 0:
                self.sleep(wait)
        return self.clock()

    def finish_frame(self, frame, started):
        finished = self.clock() - self.epoch
        arrival = frame / self.fps
        self.rows.append(dict(frame=frame, source_seconds=arrival,
                              completion_seconds=finished,
                              processing_seconds=self.clock() - started,
                              frame_age_seconds=finished - arrival if self.paced else None,
                              behind_seconds=max(0, finished - (frame + 1) / self.fps) if self.paced else None))

    def summary(self):
        if not self.rows:
            return {}
        elapsed = self.rows[-1]['completion_seconds']
        work = sum(row['processing_seconds'] for row in self.rows)
        result = dict(target_fps=self.fps, arrival_paced=self.paced, frames=len(self.rows),
                      processing_window_seconds=elapsed,
                      effective_fps=len(self.rows) / elapsed,
                      active_processing_fps=len(self.rows) / work,
                      processing_p95_seconds=percentile([r['processing_seconds'] for r in self.rows], 95),
                      definition='Frame age: completion minus scheduled arrival. Behind: completion minus the next frame deadline. Model loading precedes arrival schedule; first-inference startup is included. Every frame is retained.')
        if self.paced:
            behind = [r['behind_seconds'] for r in self.rows]
            result.update(final_behind_seconds=behind[-1], max_behind_seconds=max(behind),
                          p95_behind_seconds=percentile(behind, 95),
                          p95_frame_age_seconds=percentile([r['frame_age_seconds'] for r in self.rows], 95),
                          final_lag_frames_equivalent=math.ceil(behind[-1] * self.fps),
                          windows=[])
            for minute in sorted({int(r['source_seconds'] // 60) for r in self.rows}):
                rows = [r for r in self.rows if int(r['source_seconds'] // 60) == minute]
                result['windows'].append(dict(source_minute=minute + 1, frames=len(rows),
                    active_processing_fps=len(rows) / sum(r['processing_seconds'] for r in rows),
                    end_behind_seconds=rows[-1]['behind_seconds'],
                    p95_frame_age_seconds=percentile([r['frame_age_seconds'] for r in rows], 95)))
        return result

    def save_csv(self, path):
        with open(path, 'w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
