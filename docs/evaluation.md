# Model Evaluation

## 1. Evaluation approach

The system uses the pretrained Qwen2.5-VL-3B-Instruct model.
No model training or fine-tuning was performed.

Two videos were manually annotated and compared with model predictions. 
Frames were sampled at 2 frames per second.

Activity and bed-occupancy accuracy are calculated at sampled timestamps. 
UNKNOWN predictions are included in the calculation.

Duration estimates hold each predicted label until the next timestamp, with the final label held until the video ends.

Annotations were created during development after inspecting model outputs. 
These results are development evaluations,
not an independent held-out benchmark.

## 2. Evaluation videos

| Video       | Duration      | Samples | Manually annotated behaviour |
| room.mp4    | 19.68 seconds | 39      | Sitting upright on the bed and reading |
| room_02.mp4 | 8.12 seconds  | 16      | Sitting, standing, walking away and then leaving view |

The second video's manually annotated intervals are:

- 0.00–5.00 seconds: SITTING_ON_BED / IN_BED
- 5.00–5.50 seconds: STANDING / OUT_OF_BED
- 5.50–6.00 seconds: WALKING / OUT_OF_BED
- 6.00–8.12 seconds: UNKNOWN / UNKNOWN

These transition times are approximate. Once the person leaves
view, their current activity and occupancy are labelled UNKNOWN.

## 3. Classification results

Results below use the reviewed predictions.

| Video       | Activity accuracy | Bed occupancy accuracy |
| room.mp4    | 0.00% (0/39)      | 100.00% (39/39) |
| room_02.mp4 | 25.00% (4/16)     | 81.25% (13/16) |

For room.mp4, all 39 sitting observations were classified as LYING_IN_BED. 
Bed occupancy remained correct because both sitting and lying on the bed correspond to IN_BED.

For room_02.mp4, the four correct activity predictions consisted of one SITTING_ON_BED observation and three UNKNOWN observations.
The 25% score therefore does not demonstrate successful recognition of standing or walking.

### Nonzero activity confusion entries

| Video       | Actual label   | Predicted label  | Count |
| room.mp4    | SITTING_ON_BED | LYING_IN_BED     | 39 |
| room_02.mp4 | SITTING_ON_BED | LYING_IN_BED     | 9 |
| room_02.mp4 | SITTING_ON_BED | SITTING_ON_BED   | 1 |
| room_02.mp4 | STANDING       | SITTING_ON_BED   | 1 |
| room_02.mp4 | WALKING        | SITTING_ON_BED   | 1 |
| room_02.mp4 | UNKNOWN.       | SITTING_ON_BED   | 1 |
| room_02.mp4 | UNKNOWN        | UNKNOWN          | 3 |

## 4. Effect of context review

For the revised second-video run:

| Metric                 | Before review | After review |
| Activity accuracy.     | 0.00%         | 25.00% |
| Bed occupancy accuracy | 62.50%        | 81.25% |

Seven observations triggered review, requiring 11 additional model calls.

Although overall sample accuracy increased, review did not recover the standing and walking transition. 
Predicted out-of-bed duration fell from approximately 2.62 seconds to 0 seconds, compared with 1 second in the manual annotations.

For the first video, no observations triggered review. 
The model's clear but incorrect lying predictions were therefore retained.

## 5. Activity duration errors

### room.mp4

| Activity       | Predicted seconds | Annotated seconds | Absolute error |
| LYING_IN_BED   | 19.68             | 0.00              | 19.68 |
| SITTING_ON_BED | 0.00.             | 19.68             | 19.68 |

All remaining activities had zero predicted and annotated duration.

### room_02.mp4 — reviewed

| Activity            | Predicted seconds | Annotated seconds | Absolute error |
| LYING_IN_BED        | 4.50              | 0.00              | 4.50 |
| SITTING_ON_BED      | 2.00.             | 5.00              | 3.00 |
| SITTING_OUTSIDE_BED | 0.00.             | 0.00              | 0.00 |
| STANDING            | 0.00              | 0.50              | 0.50 |
| WALKING             | 0.00              | 0.50              | 0.50 |
| OUT_OF_BED          | 0.00              | 0.00              | 0.00 |
| UNKNOWN             | 1.62              | 2.12              | 0.50 |

## 6. Bed occupancy duration errors

| Video       | Occupancy  | Predicted seconds | Annotated seconds | Absolute error |
| room.mp4    | IN_BED     | 19.68             | 19.68             | 0.00 |
| room.mp4    | OUT_OF_BED | 0.00              | 0.00              | 0.00 |
| room.mp4    | UNKNOWN    | 0.00              | 0.00              | 0.00 |
| room_02.mp4 | IN_BED     | 6.50              | 5.00              | 1.50 |
| room_02.mp4 | OUT_OF_BED | 0.00              | 1.00              | 1.00 |
| room_02.mp4 | UNKNOWN    | 1.62              | 2.12              | 0.50 |

## 7. Bed event evaluation

Events are matched one-to-one by event type and start time, using a 1-second tolerance.

| Video       | Event         | Actual | Predicted |TP | FP| FN|
| room.mp4    | BED_EXIT      | 0      | 0         | 0 | 0 | 0 |
| room.mp4    | RETURN_TO_BED | 0      | 0         | 0 | 0 | 0 |
| room_02.mp4 | BED_EXIT      | 1      | 0         | 0 | 0 | 1 |
| room_02.mp4 | RETURN_TO_BED | 0      | 0         | 0 | 0 | 0 |

For the second video's bed exit:

- Precision: undefined because there were no predicted exits.
- Recall: 0%.
- F1: 0%.

Where there are no actual or predicted events, precision, recall and F1 are undefined, rather than evidence of perfect
performance.

The system missed the only annotated bed exit. 
No real-video return-to-bed event was available to evaluate return detection.

## 8. Monitoring outputs

| Video       | NORMAL observations | MONITOR observations | ALERT observations |
| room.mp4    |  39                 | 0                    | 0 |
| room_02.mp4 | 13                  | 3                    | 0 |

For the second video, MONITOR began at the observation timestamp 6.50 seconds because activity or occupancy was unresolved.

These are decision counts, not monitoring accuracy scores.
No manually labelled monitoring benchmark was used.

Processing is offline and may use later frames during review.
An observation timestamp is not necessarily the time at which a decision would become available in a live system.

Neither clip tests the configured 180-second absence threshold.

## 9. Failure analysis

Three observed failure cases were identified:

1. Sitting was repeatedly classified as lying, including
   observations described by the model itself as sitting.
2. The standing and walking sequence was not correctly
   recognised, preventing confirmation of the annotated exit.
3. Predictions remained incorrect around the point where the
   person left view; later unresolved observations became UNKNOWN.

Structural validation rejected incompatible activity and occupancy fields, but could not verify whether descriptions matched images.

Additional discussion is provided in failure_cases.md.

## 10. Software tests

The recorded test run passed 19 tests:

- 7 bed-event logic tests.
- 7 monitoring logic tests.
- 5 context-review tests.

These tests verify processing rules and controlled scenarios.
They do not measure visual recognition accuracy.

The evaluation scripts have not been covered by this 19-test suite.

## 11. Limitations and conclusion

- Only two short videos were evaluated.
- Sampled frames from the same video are highly correlated.
- Manual transition timestamps are approximate.
- Scene-cut verification was not completed in the annotations.
- Development examples influenced prompt changes.
- Historical results may reflect earlier prompt versions.
- No independent held-out evaluation was performed.
- Return-to-bed detection and prolonged-absence alerts remain
  untested on real video.
- Performance across people, camera positions and lighting
  conditions has not been established.

The pipeline produces predictions, summaries, events, monitoring
decisions and evaluation outputs. However, measured activity
recognition is poor and the only annotated bed exit was missed.
The current model is not validated for dependable care monitoring.

## 12. Supporting results

First video:
outputs/video_review_01/

Second video:
outputs/video_review_02_revised/

Each evaluated folder contains:
- observations.json
- evaluation.json
- bed_events.json
- event_evaluation.json
- monitoring.json

Manual annotations:
- data/annotations/room_ground_truth.json
- data/annotations/room_02_ground_truth.json