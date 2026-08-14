import unittest

from scripts.prepare_gmdcsa24_manifest import canonical_label, parse_intervals


class GmdcsaManifestParsingTest(unittest.TestCase):
    def test_fall_and_adl_intervals(self):
        parsed = parse_intervals(
            "Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]", 6.0
        )
        self.assertEqual([p["label"] for p in parsed], ["fall", "sitting"])
        self.assertEqual(parsed[0]["start_sec"], 3.4)

    def test_spacing_and_trailing_typo(self):
        parsed = parse_intervals(
            "Falling (BW) [ 0.3 to 5]:; Standing [0 to 0.2]", 5.0
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["label"], "fall")

    def test_interval_is_clipped_to_video_duration(self):
        parsed = parse_intervals("Walking[0 to 10]", 4.2)
        self.assertEqual(parsed[0]["end_sec"], 4.2)

    def test_known_label_alias(self):
        self.assertEqual(canonical_label(" Sleeping "), "lying")
        self.assertEqual(canonical_label("Fall (FW)"), "fall")


if __name__ == "__main__":
    unittest.main()
