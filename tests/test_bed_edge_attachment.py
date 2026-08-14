import unittest

import numpy as np

from bed_monitor.risk_rules import classify_seg_attachment_with_preset


def skeleton(*, seated: bool, x: float) -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    if seated:
        left = ((x, 30), (x, 50), (x - 17, 50), (x - 17, 70))
        right = ((x + 1, 30), (x + 1, 50), (x - 16, 50), (x - 16, 70))
    else:
        left = ((x, 20), (x, 40), (x, 60), (x, 80))
        right = ((x + 1, 20), (x + 1, 40), (x + 1, 60), (x + 1, 80))
    for indices, points in (((5, 11, 13, 15), left), ((6, 12, 14, 16), right)):
        for idx, point in zip(indices, points):
            xy[idx] = point
            conf[idx] = 0.95
    return xy, conf


class BedEdgeAttachmentTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((100, 100), dtype=np.uint8)
        self.mask[10:90, 50:90] = 255
        self.bed = {"mask": self.mask, "bbox": (50, 10, 89, 89)}
        self.preset = {"events": {"edge_contact_max_ratio": 0.10}}

    def classify(self, seated: bool, x: float):
        xy, conf = skeleton(seated=seated, x=x)
        center_y = 50.0 if seated else 40.0
        center = (float((xy[11, 0] + xy[12, 0]) / 2), center_y)
        return classify_seg_attachment_with_preset(
            xy, conf, center, self.bed, self.preset, conf_threshold=0.3
        )

    def test_seated_pelvis_just_outside_mask_is_partial(self):
        attachment, ratio, limbs_outside = self.classify(True, 47.0)
        self.assertEqual(attachment, "partial")
        self.assertEqual(ratio, 0.0)
        self.assertTrue(limbs_outside)

    def test_standing_beside_bed_remains_off_seg(self):
        attachment, _, _ = self.classify(False, 47.0)
        self.assertEqual(attachment, "off_seg")

    def test_seated_person_far_from_bed_remains_off_seg(self):
        attachment, _, _ = self.classify(True, 30.0)
        self.assertEqual(attachment, "off_seg")


if __name__ == "__main__":
    unittest.main()
