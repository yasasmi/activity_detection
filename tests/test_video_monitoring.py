import unittest

from activity_detection.video_monitoring import generate_monitoring


def observation(time, state, status, available=None):
    return {
        "time_sec": time,
        "state": state,
        "bed_status": status,
        "available_at_sec": time if available is None else available,
    }


def bed_exit(start=10, support=10, confirmed=10):
    return {
        "event": "bed_exit",
        "start_sec": start,
        "supporting_observation_sec": support,
        "confirmed_sec": confirmed,
    }


class MonitoringTests(unittest.TestCase):
    def test_in_bed_is_normal(self):
        observations = [
            observation(0, "SITTING_ON_BED", "IN_BED"),
        ]

        results = generate_monitoring(observations, [], 180)

        self.assertEqual(results[0]["decision"], "NORMAL")

    def test_outside_without_exit_is_monitored(self):
        observations = [
            observation(0, "STANDING", "OUT_OF_BED"),
            observation(200, "STANDING", "OUT_OF_BED"),
        ]

        results = generate_monitoring(observations, [], 180)

        self.assertEqual(results[-1]["decision"], "MONITOR")
        self.assertFalse(results[-1]["unresolved_confirmed_exit"])

    def test_alert_starts_at_threshold(self):
        observations = [
            observation(0, "LYING_IN_BED", "IN_BED"),
            observation(10, "WALKING", "OUT_OF_BED"),
            observation(189, "SITTING_OUTSIDE_BED", "OUT_OF_BED"),
            observation(190, "SITTING_OUTSIDE_BED", "OUT_OF_BED"),
        ]

        results = generate_monitoring(
            observations, [bed_exit()], 180
        )

        self.assertEqual(results[-2]["decision"], "MONITOR")
        self.assertEqual(results[-1]["decision"], "ALERT")
        self.assertEqual(
            results[-1]["continuous_predicted_out_of_bed_sec"],
            180,
        )

    def test_unknown_breaks_timer_but_retains_exit(self):
        observations = [
            observation(0, "LYING_IN_BED", "IN_BED"),
            observation(10, "WALKING", "OUT_OF_BED"),
            observation(100, "UNKNOWN", "UNKNOWN"),
            observation(200, "STANDING", "OUT_OF_BED"),
            observation(379, "STANDING", "OUT_OF_BED"),
            observation(380, "STANDING", "OUT_OF_BED"),
        ]

        results = generate_monitoring(
            observations, [bed_exit()], 180
        )

        self.assertEqual(results[2]["decision"], "MONITOR")
        self.assertTrue(results[2]["unresolved_confirmed_exit"])
        self.assertEqual(
            results[3]["continuous_predicted_out_of_bed_sec"],
            0,
        )
        self.assertEqual(results[4]["decision"], "MONITOR")
        self.assertEqual(results[5]["decision"], "ALERT")

    def test_observed_in_bed_clears_exit(self):
        observations = [
            observation(0, "LYING_IN_BED", "IN_BED"),
            observation(10, "WALKING", "OUT_OF_BED"),
            observation(20, "SITTING_ON_BED", "IN_BED"),
            observation(30, "STANDING", "OUT_OF_BED"),
        ]

        results = generate_monitoring(
            observations, [bed_exit()], 180
        )

        self.assertEqual(results[2]["decision"], "NORMAL")
        self.assertFalse(results[2]["unresolved_confirmed_exit"])
        self.assertEqual(results[3]["decision"], "MONITOR")
        self.assertFalse(results[3]["unresolved_confirmed_exit"])

    def test_future_context_does_not_inflate_absence(self):
        observations = [
            observation(0, "LYING_IN_BED", "IN_BED"),
            observation(10, "WALKING", "OUT_OF_BED", available=14),
        ]

        results = generate_monitoring(
            observations,
            [bed_exit(confirmed=14)],
            180,
        )

        self.assertEqual(
            results[-1]["decision_available_at_sec"], 14
        )
        self.assertEqual(
            results[-1]["continuous_predicted_out_of_bed_sec"], 0
        )

    def test_exit_is_not_applied_before_supporting_observation(self):
        observations = [
            observation(0, "LYING_IN_BED", "IN_BED"),
            observation(10, "STANDING", "OUT_OF_BED"),
            observation(12, "WALKING", "OUT_OF_BED"),
        ]

        results = generate_monitoring(
            observations,
            [bed_exit(start=10, support=12, confirmed=14)],
            180,
        )

        self.assertFalse(results[1]["unresolved_confirmed_exit"])
        self.assertTrue(results[2]["unresolved_confirmed_exit"])
        self.assertEqual(
            results[2]["decision_available_at_sec"], 14
        )


if __name__ == "__main__":
    unittest.main()
