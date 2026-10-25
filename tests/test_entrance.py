import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch
from experiments import retail_entrance
from experiments.retail_entrance import score_events


class EntranceScoringTests(unittest.TestCase):
    def test_duplicate_and_wrong_direction_are_extra_events(self):
        reference = [{'frame': 100, 'point': [240, 145], 'direction': 'entry'}]
        predicted = [{'frame': 102, 'point': [242, 145], 'direction': 'entry'},
                     {'frame': 103, 'point': [243, 145], 'direction': 'entry'},
                     {'frame': 101, 'point': [240, 145], 'direction': 'exit'}]
        score = score_events(reference, predicted)
        self.assertEqual((score['matched'], score['extra'], score['missed']), (1, 2, 0))

    def test_events_far_away_or_too_late_do_not_match(self):
        reference = [{'frame': 100, 'point': [240, 145], 'direction': 'entry'}]
        for event in [{'frame': 126, 'point': [240, 145], 'direction': 'entry'},
                      {'frame': 100, 'point': [280, 145], 'direction': 'entry'}]:
            score = score_events(reference, [event])
            self.assertEqual((score['matched'], score['extra'], score['missed']), (0, 1, 1))

    def test_empty_negative_clip_does_not_invent_perfect_recall(self):
        score = score_events([], [])
        self.assertIsNone(score['precision'])
        self.assertIsNone(score['recall'])

    def test_identity_audit_separates_late_count_and_duplicate(self):
        reference = [{'frame': 0, 'track_id': 7, 'point': [20, 40], 'direction': 'entry'}]
        record = {'events': [{'frame': 50, 'track_id': 'person-1', 'point': [20, 40], 'direction': 'entry'},
                             {'frame': 51, 'track_id': 'person-1', 'point': [20, 40], 'direction': 'entry'}],
                  'observations': [[['person-1', 10, 10, 30, 40, .9]] for _ in range(55)]}
        xml = '<dataset>' + ''.join(f'<frame number="{i}"><objectlist><object id="7"><box xc="20" yc="25" w="20" h="30"/></object></objectlist></frame>' for i in range(55)) + '</dataset>'
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(Path(folder) / 'annotations.zip', 'w') as archive:
                archive.writestr('sample.xml', xml)
            with patch.object(retail_entrance, 'DATA', Path(folder)), patch.object(retail_entrance, 'manifest', return_value={'entrance_references': {'sequences': {'sample': {'events': reference}}}}):
                retail_entrance.audit_events('sample', record)
        audit = record['count_audit']
        self.assertEqual(audit['identity_verified_matches'], 1)
        self.assertEqual(audit['unverified_predictions'], 1)
        self.assertEqual(audit['delayed_or_early_over_one_second'][0]['delay_seconds'], 2)


if __name__ == '__main__':
    unittest.main()
