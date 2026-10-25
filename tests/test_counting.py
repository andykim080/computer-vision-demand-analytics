import unittest
from counting import LineCounter, confirmation_frames


class CountingTests(unittest.TestCase):
    def test_confirmation_has_consistent_duration_across_fps(self):
        for fps in [5, 10, 25, 30]:
            frames = confirmation_frames({'confirm_seconds': .2}, fps)
            self.assertGreaterEqual(frames / fps, .2)
            self.assertLess(frames / fps, .2 + 1 / fps)
        self.assertEqual(confirmation_frames({'confirm_frames': 4}, 25), 4)

    def make(self, **kwargs):
        return LineCounter((10, 50), (90, 50), deadband=3, confirm_frames=2, **kwargs)

    def feed(self, counter, points, start=0, identity=1):
        return [counter.update(identity, point, start+i) for i, point in enumerate(points)]

    def test_entry_exit_and_reentry(self):
        c = self.make()
        result = self.feed(c, [(50,y) for y in [20,20,60,60,70,40,40,60,60]])
        self.assertEqual([v for v in result if v], ["entry", "exit", "entry"])

    def test_deadband_and_jitter(self):
        c = self.make()
        result = self.feed(c, [(50,y) for y in [20,20,49,51,50,52,48,60,60]])
        self.assertEqual([v for v in result if v], ["entry"])

    def test_pending_crossing_survives_destination_confirmation_jitter(self):
        c = LineCounter((0,50),(100,50),deadband=3,confirm_frames=3)
        result = self.feed(c,[(50,y) for y in [20,20,20,60,50,60,60,60]])
        self.assertEqual([e for e in result if e],['entry'])

    def test_return_to_original_side_cancels_pending_crossing(self):
        c = LineCounter((10,50),(90,50),deadband=3,confirm_frames=3)
        # Cross once without confirmation, return, then walk around the end.
        points = [(50,20)]*3+[(50,60),(50,50),(50,20)]+[(110,20),(110,60),(110,60),(110,60)]
        self.assertFalse(any(self.feed(c,points)))

    def test_confirmation_gap_keeps_crossing_but_long_gap_resets(self):
        c = LineCounter((0,50),(100,50),deadband=3,confirm_frames=3,max_gap_frames=5)
        self.feed(c,[(50,20)]*3+[(50,60)])
        self.assertEqual(self.feed(c,[(50,60)]*3,start=5)[-1],'entry')
        c = LineCounter((0,50),(100,50),deadband=3,confirm_frames=3,max_gap_frames=5)
        self.feed(c,[(50,20)]*3+[(50,60)])
        self.assertFalse(any(self.feed(c,[(50,60)]*3,start=10)))

    def test_no_crossing_beyond_endpoints(self):
        c = self.make()
        self.assertFalse(any(self.feed(c, [(100,20),(100,20),(100,80),(100,80)])))

    def test_gap_resets_history(self):
        c = self.make(max_gap_frames=3)
        self.feed(c, [(50,20),(50,20)])
        self.assertFalse(any(self.feed(c, [(50,80),(50,80)], start=10)))

    def test_initial_detection_is_not_an_entry(self):
        self.assertFalse(any(self.feed(self.make(), [(50,80)]*5)))

    def test_unconfirmed_side_change_does_not_count(self):
        c = self.make()
        self.assertFalse(any(self.feed(c, [(50,y) for y in [20,20,60,20,20]])))

    def test_reverse_direction(self):
        c = self.make(entry_side="negative")
        self.assertEqual(self.feed(c, [(50,y) for y in [20,20,60,60]])[-1], "exit")

    def test_separate_id_is_not_linked(self):
        c = self.make()
        self.feed(c, [(50,20)]*2, identity=1)
        self.assertFalse(any(self.feed(c, [(50,80)]*2, start=2, identity=2)))

    def test_diagonal_line(self):
        c = LineCounter((0,0),(100,100), deadband=1, confirm_frames=1)
        self.assertIsNone(c.update(1,(50,40),0))
        self.assertEqual(c.update(1,(50,60),1), "entry")

    def test_expired_states_pruned(self):
        c = self.make(max_gap_frames=3)
        c.update(1,(50,20),0)
        c.prune(4)
        self.assertFalse(c.states)

    def test_crossing_around_endpoint_does_not_leak(self):
        c = self.make()
        result = self.feed(c, [(50,20),(50,20),(110,60),(110,60),(50,60),(50,60)])
        self.assertFalse(any(result))


if __name__ == "__main__":
    unittest.main()
