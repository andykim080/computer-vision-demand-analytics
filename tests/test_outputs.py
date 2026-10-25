import json
from pathlib import Path
import tempfile
import unittest
from tracker import output_name, read_report, save_report


class OutputTests(unittest.TestCase):
    def test_readable_name_and_safe_rerun(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            self.assertEqual(output_name(base, 'Front Door Camera'), 'front-door-camera')
            (base / 'front-door-camera.mp4').write_bytes(b'existing video')
            save_report(base, {'runs': {'front-door-camera-2': {'events': []}}})
            self.assertEqual(output_name(base, 'Front Door Camera'), 'front-door-camera-3')
            self.assertEqual((base / 'front-door-camera.mp4').read_bytes(), b'existing video')

    def test_shared_report_keeps_previous_results(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            report = read_report(base)
            report['runs']['people'] = {'events': [{'direction': 'entry'}]}
            save_report(base, report)
            updated = read_report(base)
            updated['runs']['cars'] = {'events': []}
            save_report(base, updated)
            self.assertEqual(read_report(base)['runs'], updated['runs'])
            self.assertEqual([p.name for p in base.iterdir()], ['report.json'])

    def test_invalid_report_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / 'report.json').write_text('{broken')
            with self.assertRaises(json.JSONDecodeError):
                output_name(base, 'new-run')
            self.assertEqual((base / 'report.json').read_text(), '{broken')
