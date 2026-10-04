# Observed failure cases

## Case 1: Sitting against a headboard classified as lying

Video: data/room.mp4
Interval: 0.00–19.68 seconds
Annotation source: manual user video review

The person sits upright against the headboard, with legs extended
on the bed, while reading a black book.

Ground-truth activity: SITTING_ON_BED
Predicted activity: LYING_IN_BED
Incorrect activity predictions: 39 of 39
Activity accuracy on this clip: 0%
Bed-occupancy accuracy on this clip: 100%

The model assigned 19.68 seconds to lying instead of sitting.
Bed occupancy remained correct throughout.

The context-review controller requested no additional frames
because the predictions were consistent and marked clear.
This demonstrates that the current review triggers do not
catch every classification error.

Possible explanation:
Supported posture and extended legs may have contributed to
the confusion. This is a hypothesis, not a verified cause.

Potential improvement to test:
Clarify sitting versus lying using torso posture in the prompt,
then compare results on this development clip and separate,
previously unused clips.

Evidence:
outputs/video_review_01/evaluation.json
outputs/video_review_01/observations.json

Only one observed failure case has been documented so far.

## Case 2: Bed departure missed

Video: data/room_02.mp4
Run: outputs/video_review_02_revised
Annotation source: manual user review; approximate timestamps

Actual sequence:
- 0–5 seconds: sitting on the bed.
- 5–5.5 seconds: standing.
- 5.5–6 seconds: walking away.
- From about 6 seconds: person outside camera view.

After context review, the model labelled the observations at
5.0 and 5.5 seconds as SITTING_ON_BED. The required standing/
walking-away evidence was not established.

Actual bed exits: 1
Predicted bed exits: 0
False negatives: 1
False positives: 0
Exit recall: 0%
Exit precision: undefined because no exits were predicted.

The failure occurred in the complete perception-to-event pipeline.
This result alone does not establish that the event rules are wrong.

Evidence:
outputs/video_review_02_revised/evaluation.json
outputs/video_review_02_revised/bed_events.json
outputs/video_review_02_revised/event_evaluation.json


## Case 3: Activity predicted after the person leaves view

Video: data/room_02.mp4
Annotation source: manual user review; disappearance around 6 seconds.

Ground truth from approximately 6 seconds onward: UNKNOWN.

Before review:
- 6.0 and 6.5 seconds: STANDING.
- 7.0 and 7.5 seconds: OUT_OF_BED.

After review:
- 6.0 seconds: SITTING_ON_BED.
- 6.5, 7.0 and 7.5 seconds: UNKNOWN.

Review corrected three later observations but did not resolve
the observation at the approximate disappearance boundary.

Earlier context may have influenced the target-frame prediction,
but this cause has not been established. The approximate manual
boundary also introduces timing uncertainty.

Evidence:
outputs/video_run_02_revised/evaluation.json
outputs/video_review_02_revised/evaluation.json


## Evaluation limitations

These three cases describe failures across two videos, not three
independent videos. Both clips were inspected during development,
and prompts were revised using examples from the second clip.

Results are development-set measurements, not held-out estimates
of general performance.

No real-video return-to-bed event has been evaluated.
No real-video prolonged-absence alert has been evaluated.
Passing unit tests does not demonstrate model accuracy.
