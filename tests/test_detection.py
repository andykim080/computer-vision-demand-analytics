import unittest
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from ultralytics.engine.results import Boxes
from detection import merge_people, person_keep_indices
from tracker import SeparateVehicleTracker


class DetectionTests(unittest.TestCase):
    def test_nested_partial_duplicate_keeps_full_body_and_offset_people(self):
        rows = [[10,10,40,90,.7,0], [15,11,35,35,.95,0],
                [25,15,55,95,.8,0], [15,40,35,80,.8,0]]
        # High-confidence duplicate head is removed; neighboring person and a
        # smaller body with a different top remain, including stationary people.
        self.assertEqual(person_keep_indices(rows, {'nested_height_ratio': .7}), [0,2,3])

    def test_person_nms_is_optional_and_does_not_modify_input(self):
        rows = [[10,10,40,90,.7,0],[11,10,41,90,.8,0]]
        self.assertEqual(person_keep_indices(rows, None), [0,1])
        self.assertEqual(person_keep_indices(rows, {'iou': .6}), [1])
        self.assertEqual(rows[0][0], 10)
        self.assertEqual(person_keep_indices([], {'iou': .6}), [])

    def test_camera_local_person_suppression_preserves_overlapping_vehicle(self):
        tracker = SeparateVehicleTracker(Path(__file__).resolve().parents[1] / 'configs/bytetrack-low-fps.yaml',
            {'track_high_thresh': .2, 'new_track_thresh': .2},
            person_suppression={'nested_height_ratio': .7})
        rows = np.array([[10,10,40,90,.8,0], [15,11,35,35,.95,0],
                         [10,10,40,90,.8,2]], dtype=np.float32)
        result = tracker.update(SimpleNamespace(boxes=Boxes(rows,(100,100))),
                                np.zeros((100,100,3),np.uint8),{0:'person',2:'car'},.1,.7)
        self.assertEqual(sorted(row[2] for row in result),['car','person'])
        self.assertEqual(tracker.person_boxes_suppressed,1)

    def test_person_views_merge_duplicates_and_keep_overlapping_vehicle(self):
        original = [[10, 10, 30, 60, .8, 0], [10, 10, 30, 60, .7, 3]]
        extra = [[11, 10, 31, 60, .9, 0], [40, 10, 60, 60, .8, 0]]
        rows = merge_people(original, extra, .1)
        self.assertEqual(list(rows[:, 5]).count(0), 2)
        self.assertEqual(list(rows[:, 5]).count(3), 1)
        self.assertEqual(original[0][0], 10)

    def test_empty_person_passes_preserve_vehicles(self):
        rows = merge_people([[1, 2, 3, 4, .8, 2]], [], .1)
        self.assertEqual(rows.shape, (1, 6))
        self.assertEqual(merge_people([], [], .1).shape, (0, 6))

    def test_tiny_weak_vehicle_filter_keeps_people_and_real_vehicle_cases(self):
        tracker = SeparateVehicleTracker(Path(__file__).resolve().parents[1] / 'configs/bytetrack-low-fps.yaml',
            {'track_high_thresh': .2, 'new_track_thresh': .2,
             'small_box_height_fraction': .3, 'small_box_min_conf': .35})
        # Tiny weak pillar, tiny strong car, large weak parked car, small person.
        rows = np.array([[1,1,20,20,.24,2],[30,1,50,20,.8,2],
                         [60,1,90,60,.25,2],[1,50,10,70,.8,0]], dtype=np.float32)
        result = tracker.update(SimpleNamespace(boxes=Boxes(rows, (100,100))),
                                np.zeros((100,100,3),np.uint8), {0:'person',2:'car'}, .1, .7)
        self.assertEqual(sorted(r[2] for r in result), ['car','car','person'])
        self.assertTrue(all(r[0][0] > 20 or r[2] == 'person' for r in result))

    def test_person_threshold_override_is_camera_local(self):
        path = Path(__file__).resolve().parents[1] / 'configs/bytetrack-low-fps.yaml'
        settings = {'track_high_thresh': .2, 'new_track_thresh': .2}
        dark = SeparateVehicleTracker(path, settings, {'new_track_thresh': .45})
        default = SeparateVehicleTracker(path, settings)
        self.assertEqual(dark.settings['person']['new_track_thresh'], .45)
        self.assertEqual(default.settings['person']['new_track_thresh'], .55)

    def test_track_expiration_uses_seconds_only_when_configured(self):
        path = Path(__file__).resolve().parents[1] / 'configs/bytetrack-low-fps.yaml'
        settings = {'track_high_thresh': .2, 'new_track_thresh': .2}
        for fps in [5, 25, 30]:
            tracker = SeparateVehicleTracker(path, settings, frame_rate=fps, track_buffer_seconds=1.2)
            self.assertAlmostEqual(tracker.trackers['person'].max_frames_lost / fps, 1.2)
        legacy = SeparateVehicleTracker(path, settings)
        self.assertEqual(legacy.trackers['person'].max_frames_lost, 30)
