import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from ultralytics.engine.results import Boxes

from detection import class_confidence_indices
from tracker import SeparateVehicleTracker, load_config

ROOT=Path(__file__).resolve().parents[1]
NAMES={0:'person',2:'car',3:'motorcycle',7:'truck'}


class ClassConfidenceTests(unittest.TestCase):
    def test_requested_cutoffs_and_boundary_keep_other_classes(self):
        rows=[[0,0,10,10,.54,2],[0,0,10,10,.55,2],[0,0,10,10,.54,7],
              [0,0,10,10,.2,3],[0,0,10,10,.2,0]]
        original=copy.deepcopy(rows)
        self.assertEqual(class_confidence_indices(rows,NAMES,{'car':.55,'truck':.55}),[1,3,4])
        self.assertEqual(class_confidence_indices(rows,NAMES,None),list(range(5)))
        self.assertEqual(rows,original)

    def test_low_confidence_person_and_motorcycle_recovery_remains(self):
        tracker=SeparateVehicleTracker(ROOT/'configs/bytetrack-low-fps.yaml',
            {'track_high_thresh':.2,'new_track_thresh':.2},
            class_confidence={'car':.55,'truck':.55,'person':.1,'motorcycle':.1})
        image=np.zeros((100,100,3),np.uint8)
        def update(conf):
            rows=np.array([[0,0,20,60,conf,0],[30,0,50,60,conf,3],
                           [60,0,80,60,.5,2],[80,0,99,60,.5,7]],np.float32)
            return tracker.update(SimpleNamespace(boxes=Boxes(rows,(100,100))),image,NAMES,.1,.7)
        first=update(.8);second=update(.15)
        self.assertEqual({r[2] for r in first},{'person','motorcycle'})
        self.assertEqual({r[1] for r in first},{r[1] for r in second})
        self.assertEqual(tracker.class_boxes_filtered,{'car':2,'truck':2})

    def test_invalid_thresholds_and_missing_separate_tracker_are_rejected(self):
        base=json.loads((ROOT/'configs/retail-baseline.json').read_text())
        for value in [{},[],{'car':True},{'car':0},{'car':float('nan')},{'car':1.1}]:
            cfg=copy.deepcopy(base);cfg['class_confidence']=value
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'camera.json';path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):load_config(path)
        cfg=copy.deepcopy(base);cfg.pop('vehicle_tracking');cfg['class_confidence']={'car':.55}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'camera.json';path.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load_config(path)


if __name__=='__main__':unittest.main()
