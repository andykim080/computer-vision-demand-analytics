from pathlib import Path
import unittest
import numpy as np
from ultralytics.engine.results import Boxes
from tracker import SeparateVehicleTracker
from counting import LineCounter

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: 'person', 2: 'car', 3: 'motorcycle', 7: 'truck'}


class VehicleTrackingTests(unittest.TestCase):
    def tracker(self):
        return SeparateVehicleTracker(ROOT / 'configs/bytetrack-low-fps.yaml',
                                      {'track_high_thresh': .2, 'new_track_thresh': .2})

    def update(self, tracker, rows):
        from types import SimpleNamespace
        return tracker.update(SimpleNamespace(boxes=Boxes(np.asarray(rows,dtype=np.float32).reshape(-1,6), (100,100))),
                              np.zeros((100,100,3),np.uint8), NAMES, .1, .7)

    def test_vehicles_start_early_without_lowering_person_gate(self):
        tracker = self.tracker()
        rows = self.update(tracker, [[10,10,30,60,.49,0],[50,10,80,60,.35,2]])
        self.assertEqual([row[2] for row in rows], ['car'])
        self.assertEqual(tracker.settings['person']['new_track_thresh'], .55)
        self.assertEqual(tracker.settings['vehicle']['new_track_thresh'], .2)

    def test_duplicate_vehicle_classes_do_not_remove_overlapping_person(self):
        rows = self.update(self.tracker(), [[10,10,30,60,.8,2],[10,10,30,60,.7,7],[10,10,30,60,.9,0]])
        self.assertEqual(sorted(row[2] for row in rows), ['car','person'])
        self.assertEqual(len({row[1] for row in rows}),2)

    def test_motorcycle_label_can_correct_first_frame_car_guess(self):
        tracker = self.tracker()
        first = self.update(tracker, [[10,10,30,60,.24,2]])
        second = self.update(tracker, [[12,10,32,60,.8,3]])
        self.assertEqual(first[0][1],second[0][1])
        self.assertEqual(second[0][2],'motorcycle')

    def test_stationary_real_person_kept_at_low_recovery_score(self):
        tracker = self.tracker()
        first = self.update(tracker, [[10,10,30,60,.8,0]])
        second = self.update(tracker, [[10,10,30,60,.2,0]])
        self.assertEqual(first[0][1],second[0][1])
        self.assertEqual(second[0][2],'person')

    def test_short_initial_approach_still_requires_destination_confirmation(self):
        counter = LineCounter((50,0),(50,100),deadband=2,confirm_frames=3,
                              initial_confirm_frames=1)
        self.assertIsNone(counter.update('vehicle-1',(40,50),0))
        self.assertIsNone(counter.update('vehicle-1',(60,50),1))
        self.assertIsNone(counter.update('vehicle-1',(65,50),2))
        self.assertEqual(counter.update('vehicle-1',(70,50),3),'exit')

    def test_first_appearance_never_counts_as_a_crossing(self):
        counter = LineCounter((50,0),(50,100),confirm_frames=3,initial_confirm_frames=1)
        self.assertFalse(any(counter.update(1,(70,50),frame) for frame in range(5)))

    def test_quick_return_before_confirmation_is_not_counted(self):
        counter = LineCounter((50,0),(50,100),confirm_frames=3,initial_confirm_frames=1)
        self.assertFalse(any(counter.update(1,point,frame) for frame,point in enumerate([(30,50),(70,50),(30,50),(30,50)])))

    def test_short_initial_confirmation_does_not_bridge_a_long_gap(self):
        counter = LineCounter((50,0),(50,100),confirm_frames=3,
                              initial_confirm_frames=1,max_gap_frames=2)
        self.assertIsNone(counter.update(1,(30,50),0))
        self.assertIsNone(counter.update(1,(70,50),10))
        self.assertIsNone(counter.update(1,(70,50),11))
        self.assertIsNone(counter.update(1,(70,50),12))

    def test_vehicle_near_line_approach_counts_with_smaller_buffer(self):
        # Recorded clip 4: first confirmed bike is only ~4px left of the line.
        positions = [(1070,362),(1087,353),(1105,341),(1123,333)]
        def events(deadband):
            counter=LineCounter((1074,216),(1074,575),deadband=deadband,
                                confirm_frames=3,initial_confirm_frames=1,max_gap_frames=2)
            return [event for frame,point in enumerate(positions)
                    if (event:=counter.update('bike',point,frame))]
        self.assertEqual(events(8),[])
        self.assertEqual(events(3),['exit'])

    def test_smaller_vehicle_buffer_still_ignores_near_line_jitter(self):
        counter=LineCounter((1074,216),(1074,575),deadband=3,
                            confirm_frames=3,initial_confirm_frames=1,max_gap_frames=2)
        self.assertFalse(any(counter.update('bike',(x,350),frame)
                             for frame,x in enumerate([1070,1073,1076,1072,1075,1071,1074])))
