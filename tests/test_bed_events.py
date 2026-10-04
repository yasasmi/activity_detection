import unittest

from activity_detection.bed_events import detect_bed_events


def interval(start, end, state):
    """Create one activity interval for a synthetic test."""
    return {
        "start_sec": start,
        "end_sec": end,
        "state": state,
    }


class BedEventTests(unittest.TestCase):

    def test_sitting_up_is_not_an_exit(self):
        timeline = [
            interval(0, 10, "LYING_IN_BED"),
            interval(10, 20, "SITTING_ON_BED"),
            interval(20, 30, "LYING_IN_BED"),
        ]

        events = detect_bed_events(timeline, {})

        self.assertEqual(events, [])

    def test_brief_standing_is_not_an_exit(self):
        timeline = [
            interval(0, 10, "SITTING_ON_BED"),
            interval(10, 12, "STANDING"),
            interval(12, 20, "SITTING_ON_BED"),
        ]

        events = detect_bed_events(timeline, {})

        self.assertEqual(events, [])

    def test_walking_away_confirms_only_one_exit(self):
        timeline = [
            interval(0, 10, "SITTING_ON_BED"),
            interval(10, 15, "STANDING"),
            interval(15, 20, "WALKING"),
            interval(20, 25, "WALKING"),
        ]

        evidence = {
            15: {"moving_away": True},
            20: {"moving_away": True},
        }

        events = detect_bed_events(timeline, evidence)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "bed_exit")
        self.assertEqual(events[0]["start_sec"], 10)

    def test_missing_movement_evidence_does_not_confirm_exit(self):
        timeline = [
            interval(0, 10, "SITTING_ON_BED"),
            interval(10, 15, "STANDING"),
            interval(15, 25, "WALKING"),
        ]

        events = detect_bed_events(timeline, {})

        self.assertEqual(events, [])

    def test_unknown_breaks_exit_evidence(self):
        timeline = [
            interval(0, 10, "LYING_IN_BED"),
            interval(10, 20, "UNKNOWN"),
            interval(20, 30, "WALKING"),
        ]

        evidence = {
            20: {"moving_away": True},
        }

        events = detect_bed_events(timeline, evidence)

        self.assertEqual(events, [])

    def test_full_return_sequence(self):
        timeline = [
            interval(0, 10, "WALKING"),
            interval(10, 15, "SITTING_ON_BED"),
            interval(15, 25, "LYING_IN_BED"),
        ]

        evidence = {
            0: {"approaching_bed": True},
        }

        events = detect_bed_events(timeline, evidence)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "return_to_bed")
        self.assertEqual(events[0]["start_sec"], 10)

    def test_sitting_on_bed_does_not_complete_full_return(self):
        timeline = [
            interval(0, 10, "WALKING"),
            interval(10, 20, "SITTING_ON_BED"),
        ]

        evidence = {
            0: {"approaching_bed": True},
        }

        events = detect_bed_events(timeline, evidence)

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
