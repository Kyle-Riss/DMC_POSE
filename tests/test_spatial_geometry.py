import unittest

import numpy as np

from spatial_geometry import (
    orient_bed_detection,
    refined_bed_from_mask,
    select_refined_bed_candidate,
    skeleton_bed_coverage,
)


class SpatialGeometryTests(unittest.TestCase):
    def test_rotates_native_bed_mask_into_analysis_coordinates(self):
        mask = np.zeros((3, 5), dtype=np.uint8)
        mask[1:3, 1:4] = 1
        result = orient_bed_detection(
            {"mask": mask, "bbox": (1, 1, 3, 2)}, 90, 3, 5
        )
        expected = np.rot90(mask, k=3)
        np.testing.assert_array_equal(result["mask"], expected)
        self.assertEqual(result["bbox"], (0, 1, 1, 3))

    def test_all_supported_rotations_preserve_mask_area(self):
        mask = np.zeros((7, 11), dtype=np.uint8)
        mask[2:6, 3:9] = 1
        for rotation in (0, 90, 180, 270):
            result = orient_bed_detection(
                {"mask": mask, "bbox": (3, 2, 8, 5)}, rotation, 7, 11
            )
            self.assertEqual(int(result["mask"].sum()), int(mask.sum()))
            self.assertIsNotNone(result["bbox"])

    def test_skeleton_does_not_count_empty_bbox_area_as_body(self):
        bed_mask = np.zeros((100, 100), dtype=np.uint8)
        bed_mask[:, 50:] = 1
        points = np.asarray([[10, 10], [20, 30], [30, 50], [40, 70]], dtype=float)
        confidence = np.ones(4, dtype=float)
        ratio = skeleton_bed_coverage(
            points, confidence, {"mask": bed_mask}, 100, 100
        )
        self.assertEqual(ratio, 0.0)

    def test_skeleton_coverage_is_observed_joint_fraction(self):
        bed_mask = np.zeros((100, 100), dtype=np.uint8)
        bed_mask[:, 50:] = 1
        points = np.asarray([[10, 10], [55, 30], [70, 50], [40, 70]], dtype=float)
        confidence = np.ones(4, dtype=float)
        ratio = skeleton_bed_coverage(
            points, confidence, {"mask": bed_mask}, 100, 100
        )
        self.assertEqual(ratio, 0.5)

    def test_refined_mask_must_contain_coarse_center(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:40, 10:40] = 1
        result = refined_bed_from_mask(mask, (20, 20, 80, 80), 0.9, 100, 100)
        self.assertIsNone(result)

    def test_refined_mask_accepts_plausible_center_component(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:80, 20:90] = 1
        result = refined_bed_from_mask(mask, (10, 20, 90, 90), 0.9, 100, 100)
        self.assertIsNotNone(result)
        self.assertEqual(result['bbox'], (20, 30, 89, 79))
        self.assertEqual(result['source'], 'mobile_sam_multipoint_refined')

    def test_refined_mask_rejects_implausibly_large_area(self):
        mask = np.ones((100, 100), dtype=np.uint8)
        result = refined_bed_from_mask(mask, (10, 10, 90, 90), 0.9, 100, 100)
        self.assertIsNone(result)

    def test_refined_mask_rejects_small_prompted_fragment(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[35:55, 35:55] = 1
        result = refined_bed_from_mask(
            mask, (10, 10, 90, 90), 0.9, 100, 100,
            prompt_point=(45, 45),
        )
        self.assertIsNone(result)

    def test_largest_valid_prompt_candidate_is_selected(self):
        fragment = np.zeros((100, 100), dtype=np.uint8)
        fragment[35:55, 35:55] = 1
        mattress = np.zeros((100, 100), dtype=np.uint8)
        mattress[25:80, 20:90] = 1
        result = select_refined_bed_candidate(
            [fragment, mattress], [(45, 45), (65, 50)],
            (10, 10, 90, 90), 0.9, 100, 100,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["bbox"], (20, 25, 89, 79))


if __name__ == "__main__":
    unittest.main()
