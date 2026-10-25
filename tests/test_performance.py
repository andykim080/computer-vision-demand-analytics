import unittest
from performance import PlaybackTiming


class FakeClock:
    def __init__(self):
        self.now = 0
    def __call__(self):
        return self.now
    def sleep(self, duration):
        self.now += duration


class PlaybackTimingTests(unittest.TestCase):
    def test_slow_processing_accumulates_backlog_without_dropping_frames(self):
        clock = FakeClock()
        timing = PlaybackTiming(25, True, clock, clock.sleep)
        for frame in range(100):
            started = timing.begin_frame(frame)
            clock.now += .06
            timing.finish_frame(frame, started)
        result = timing.summary()
        self.assertEqual(result['frames'], 100)
        self.assertAlmostEqual(result['final_behind_seconds'], 2)
        self.assertAlmostEqual(result['effective_fps'], 100 / 6)

    def test_fast_processing_waits_for_arrivals_and_recovers_after_a_stall(self):
        clock = FakeClock()
        timing = PlaybackTiming(25, True, clock, clock.sleep)
        for frame in range(100):
            started = timing.begin_frame(frame)
            clock.now += .3 if frame == 0 else .01
            timing.finish_frame(frame, started)
        result = timing.summary()
        self.assertEqual(result['final_behind_seconds'], 0)
        self.assertGreater(result['max_behind_seconds'], .2)
        self.assertAlmostEqual(timing.rows[-1]['frame_age_seconds'], .01)

    def test_unpaced_offline_processing_does_not_claim_live_latency(self):
        clock = FakeClock()
        timing = PlaybackTiming(25, False, clock, clock.sleep)
        started = timing.begin_frame(0)
        clock.now = .01
        timing.finish_frame(0, started)
        self.assertNotIn('final_behind_seconds', timing.summary())
        self.assertIsNone(timing.rows[0]['frame_age_seconds'])


if __name__ == '__main__':
    unittest.main()
