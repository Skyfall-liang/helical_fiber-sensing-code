import json
from argparse import Namespace
from unittest import TestCase

import numpy as np

from helical_fiber import Config, HelicalFiberModel
from live import sensing_coordinates


class LiveTests(TestCase):
    def setUp(self):
        self.args = Namespace(fiber_length=None, assume_span_at_base=False, sensor_start_m=None)
        self.metadata = json.loads('''{"sensors": [{"channel": 1, "gage pitch (mm)": 2.6,
          "segments": [{"segment name": "All Gages", "index": 4, "size": 32}]}]}''')

    def test_use_segment_samples_without_rescaling(self):
        config = Config(fiber_length_m=.0806)
        crop, s = sensing_coordinates(self.metadata, 1, self.args, config, 36)
        self.assertEqual(crop, slice(4, 36))
        np.testing.assert_allclose(np.diff(s), .0026)
        self.assertAlmostEqual(s[-1], config.fiber_length_m)

    def test_refuse_to_stretch_short_span_to_paper_geometry(self):
        with self.assertRaisesRegex(ValueError, r'80.6 mm.*4277 mm'):
            sensing_coordinates(self.metadata, 1, self.args, Config(), 36)
        with self.assertRaisesRegex(ValueError, 'outside the measurement'):
            sensing_coordinates(self.metadata, 1, self.args, Config(fiber_length_m=.0806), 35)

    def test_backbone_and_contact_surface_share_base_frame(self):
        model = HelicalFiberModel()
        s = np.linspace(0, model.config.fiber_length_m, 20)
        center = model.backbone_xyz(s, 2.0, .7)
        np.testing.assert_array_equal(center[0], [0, 0, 0])
        np.testing.assert_allclose(np.linalg.norm(model.helix_xyz(s, 2.0, .7) - center, axis=1),
                                   model.config.radius_m)

    def test_explicit_partial_window_keeps_physical_pitch(self):
        self.args.assume_span_at_base = True
        crop, s = sensing_coordinates(self.metadata, 1, self.args, Config(), 36)
        self.assertEqual(crop, slice(4, 36))
        np.testing.assert_allclose(s, np.arange(32) * .0026)

    def test_sensor_origin_preserves_first_sample_offset_and_missing_tail(self):
        self.args.sensor_start_m = 1.2
        self.metadata['sensors'][0]['segments'][0].update(index=2, size=2061, **{'location (mm)': 80.0})
        crop, s = sensing_coordinates(self.metadata, 1, self.args, Config(), 2063)
        self.assertEqual(crop, slice(433, 2063))
        self.assertAlmostEqual(s[0], .0006)
        self.assertAlmostEqual(s[-1], 4.236)
        np.testing.assert_allclose(np.diff(s), .0026)
