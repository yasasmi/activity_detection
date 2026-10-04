import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from activity_detection import review_video
def clear_observation():
    return {
        "state": "SITTING_ON_BED",
        "bed_status": "IN_BED",
        "person_visible": True,
        "evidence_quality": "clear",
        "evidence": "Person is sitting supported by the bed.",
        "moving_away": False,
        "approaching_bed": False,
        "movement_evidence": "No directional movement observed.",
    }


def uncertain_observation():
    return {
        "state": "UNKNOWN",
        "bed_status": "UNKNOWN",
        "person_visible": True,
        "evidence_quality": "insufficient",
        "evidence": "Posture is obscured.",
        "moving_away": None,
        "approaching_bed": None,
        "movement_evidence": "Direction cannot be established.",
    }


class ContextReviewTests(unittest.TestCase):
    def run_review(self, initial, responses):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            predictions = folder / "predictions.json"
            manifest = folder / "manifest.json"
            output = folder / "review"

            initial = dict(initial)
            initial.update({
                "time_sec": 0.0,
                "validation_passed": True,
                "validation_error": None,
            })

            predictions.write_text(json.dumps({
                "source_video": "synthetic_test.mp4",
                "observation_duration_sec": 0.5,
                "sample_fps": 2.0,
                "observations": [initial],
            }), encoding="utf-8")

            manifest.write_text(json.dumps({
                "source_video": "synthetic_test.mp4",
                "frames": [
                    {"time_sec": 0.0, "file": "unused.jpg"}
                ],
            }), encoding="utf-8")

            arguments = [
                "review_video.py",
                "--predictions", str(predictions),
                "--manifest", str(manifest),
                "--out", str(output),
            ]

            # Replace model loading and inference with controlled responses.
            with (
                patch("sys.argv", arguments),
                patch(
                    "activity_detection.review_video.load_model",
                    return_value=(object(), object()),
                ) as loader,
                patch(
                    "activity_detection.review_video.analyze_window",
                    side_effect=responses,
                ) as inference,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                review_video.main()

            saved = json.loads(
                (output / "observations.json").read_text(
                    encoding="utf-8"
                )
            )
            log = json.loads(
                (output / "review_log.jsonl").read_text(
                    encoding="utf-8"
                ).strip()
            )

            return (
                saved["observations"][0],
                log,
                loader.call_count,
                inference.call_count,
            )

    def test_clear_in_bed_observation_skips_model(self):
        result, log, loads, calls = self.run_review(
            clear_observation(), []
        )

        self.assertEqual(loads, 0)
        self.assertEqual(calls, 0)
        self.assertEqual(result["review_attempts"], 0)
        self.assertEqual(log["attempts"], [])

    def test_resolved_first_attempt_stops(self):
        response = (clear_observation(), "mock clear response", None)

        result, log, loads, calls = self.run_review(
            uncertain_observation(), [response]
        )

        self.assertEqual(loads, 1)
        self.assertEqual(calls, 1)
        self.assertEqual(result["state"], "SITTING_ON_BED")
        self.assertTrue(log["attempts"][0]["resolved"])

    def test_second_attempt_can_resolve(self):
        responses = [
            (uncertain_observation(), "mock uncertain response", None),
            (clear_observation(), "mock clear response", None),
        ]

        result, log, loads, calls = self.run_review(
            uncertain_observation(), responses
        )

        self.assertEqual(loads, 1)
        self.assertEqual(calls, 2)
        self.assertEqual(result["state"], "SITTING_ON_BED")
        self.assertFalse(log["attempts"][0]["resolved"])
        self.assertTrue(log["attempts"][1]["resolved"])

    def test_exhausted_retries_produce_unknown(self):
        responses = [
            (uncertain_observation(), "mock response one", None),
            (uncertain_observation(), "mock response two", None),
        ]

        result, log, loads, calls = self.run_review(
            uncertain_observation(), responses
        )

        self.assertEqual(calls, 2)
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertEqual(result["bed_status"], "UNKNOWN")
        self.assertFalse(result["validation_passed"])
        self.assertEqual(
            result["validation_error"],
            "context_review_unresolved",
        )

    def test_context_includes_target_and_respects_limit(self):
        frames = [
            {"time_sec": index * 0.5, "file": f"{index}.jpg"}
            for index in range(30)
        ]

        selected = review_video.choose_context(
            frames, target_index=15, radius=4
        )
        times = [frame["time_sec"] for frame in selected]

        self.assertIn(7.5, times)
        self.assertLessEqual(len(selected), review_video.MAX_FRAMES)
        self.assertEqual(times, sorted(times))
        self.assertTrue(all(abs(time - 7.5) <= 4 for time in times))


if __name__ == "__main__":
    unittest.main()
