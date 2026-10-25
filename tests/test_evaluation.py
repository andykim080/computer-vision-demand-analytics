import unittest
import importlib.util
from evaluation import tracking_metrics,reference_crossings,crossing_metrics,ignore_fraction

class EvaluationTests(unittest.TestCase):
 @unittest.skipUnless(importlib.util.find_spec('trackeval'), 'Install the optional evaluation dependency')
 def test_perfect_and_false_track_metrics(self):
  gt=[[[1,10,10,30,60,1,1,1]]] * 4
  perfect=[[[7,10,10,30,60,.9,0]]] * 4
  s,_=tracking_metrics(gt,perfect,'person');self.assertAlmostEqual(s['HOTA'],100);self.assertAlmostEqual(s['IDF1'],100)
  extra=[frame+[[9,70,10,90,60,.9,0]] for frame in perfect]
  s,_=tracking_metrics(gt,extra,'person');self.assertEqual(s['false_detections'],4);self.assertAlmostEqual(s['precision'],50)
 @unittest.skipUnless(importlib.util.find_spec('trackeval'), 'Install the optional evaluation dependency')
 def test_official_mot_distractor_is_ignored(self):
  gt=[[[1,10,10,30,60,1,1,1],[2,70,10,90,60,1,8,1]]]
  pred=[[[7,10,10,30,60,.9,0],[9,70,10,90,60,.9,0]]]
  s,_=tracking_metrics(gt,pred,'person');self.assertEqual(s['false_detections'],0);self.assertEqual(s['labeled_boxes'],1)
 @unittest.skipUnless(importlib.util.find_spec('trackeval'), 'Install the optional evaluation dependency')
 def test_vehicle_ignore_union_and_real_target_preserved(self):
  self.assertAlmostEqual(ignore_fraction([0,0,10,10],[[0,0,4,10],[4,0,8,10]]),.8)
  s,_=tracking_metrics([[[1,0,0,10,10,1,1,1]]],[[[7,0,0,10,10,.8,2],[9,20,0,30,10,.8,2]]],'vehicle',[[0,0,10,10],[20,0,30,10]])
  self.assertEqual(s['matched_labeled_boxes'],1);self.assertEqual(s['false_detections'],0)
 def test_reference_requires_real_crossing_and_confirmation(self):
  gt=[[[1,x-2,10,x+2,30,1,1,1]] for x in [20,20,20,60,60,60]]
  r=reference_crossings(gt,[(50,0),(50,100)],5,'person',list(range(6)),100,100)
  self.assertEqual(len(r),1);self.assertEqual(r[0]['direction'],'exit')
  self.assertEqual(reference_crossings(gt[3:],[(50,0),(50,100)],5,'person',[0,1,2],100,100),[])
 def test_event_matching_penalizes_duplicate_and_missed_events(self):
  r=[dict(frame=5,point=[50,30],direction='exit',reliable=True)]
  p=[dict(frame=5,point=[50,30],direction='exit')]*2
  m=crossing_metrics(r,p,5,100);self.assertEqual(m['matched_events'],1);self.assertEqual(m['false_events'],1)
  self.assertEqual(crossing_metrics(r,[],5,100)['missed_events'],1)
