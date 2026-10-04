# Video Activity Recognition and Bed-Event Monitoring

## Overview

This project is an offline prototype for recognizing activities, estimating bed occupancy, detecting bed exits and returns, and generating explained monitoring decisions from indoor video.

It uses the pretrained Qwen2.5-VL-3B-Instruct vision-language model for visual observations and Python rules for temporal processing.

No model training or fine-tuning was performed.

The pipeline runs end to end, but evaluation identified substantial recognition errors. 
It has not been validated for real-world care use.

## Supported activities

| Label               | Meaning |
| LYING_IN_BED        | Lying supported by the bed |
| SITTING_ON_BED      | Sitting supported by the bed |
| SITTING_OUTSIDE_BED | Sitting on a surface outside the bed |
| STANDING            | Upright on feet on the floor |
| WALKING             | Visible stepping and movement across frames |
| OUT_OF_BED          | Clearly outside the bed, specific activity unclear |
| UNKNOWN             | Insufficient evidence to determine activity |

Bed occupancy is represented separately as IN_BED, OUT_OF_BED or UNKNOWN.

## Architecture

The processing stages are:

1. Extract timestamped frames using FFmpeg.
2. Generate structured visual observations using Qwen.
3. Validate labels, types and activity/occupancy consistency.
4. Request additional temporal context when review conditions apply.
5. Build activity and bed-occupancy timelines.
6. Detect meaningful bed events using temporal rules and movement evidence.
7. Generate NORMAL, MONITOR or ALERT decisions with reasons.
8. Compare predictions against manual annotations.

The Python controller manages context retrieval, bounded retries and event memory. Qwen supplies visual observations.

## Project structure

| Location            | Purpose |
| activity_detection/ | Main Python package |
| tests/              | Event, context-review and monitoring tests |
| tools/              | Environment and model setup checks |
| data/               | Input videos and images |
| data/annotations/   | Manual ground-truth JSON files |
| outputs/            | Predictions, summaries, review logs and evaluation |
| docs/               | Supporting documentation and failure analysis |
| requirements.txt    | Installed Python dependency versions |

### Main modules

| File                  | Responsibility |
| load_model.py         | Load Qwen and its processor on Apple MPS |
| prepare_video.py      | Extract frames and create a timestamp manifest |
| analyze_image.py      | Generate a descriptive single-image response |
| recognize_activity.py | Classify a single image and validate observations |
| analyze_sequence.py   | Test a four-frame sequence |
| analyze_video.py      | Generate observations across a video |
| review_video.py       | Request additional context and resolve uncertainty |
| activity_timeline.py  | Build activity intervals and calculate durations |
| summarize_video.py    | Summarize real-video activity predictions |
| summarize_bed.py.     | Summarize predicted bed occupancy |
| bed_events.py         | Apply temporal exit and return rules |
| video_events.py       | Connect reviewed observations to event detection |
| video_monitoring.py   | Generate real-video monitoring decisions |
| evaluate_video.py     | Calculate classification and duration metrics |
| evaluate_events.py    | Calculate event matching and detection metrics |

bed_occupancy.py and monitoring.py contain the earlier synthetic demonstration logic. The real-video commands use summarize_bed.py and video_monitoring.py.

## Development environment

Developed and run on:

- Apple M5 MacBook Air with 16 GB memory.
- Python 3.11.
- PyTorch with Apple MPS acceleration.
- float16 model weights.
- Transformers and qwen-vl-utils.
- Pillow and Accelerate.
- FFmpeg and FFprobe.

The current model loader requires MPS. CPU and CUDA execution have not been configured or validated.

Exact installed Python package versions are recorded in requirements.txt.

## Installation

Run commands from the outer project folder, which contains the activity_detection package directory.

For a fresh environment on a compatible Apple Silicon Mac:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

FFmpeg and FFprobe must also be installed and available on PATH.

Check the environment:

```bash
python -m tools.check_setup
python -m tools.check_model_setup
```

The first inference run downloads the model weights if they are not already cached. An internet connection is needed for that download.

For an existing environment, activate it before running commands:

```bash
source .venv/bin/activate
```

## Run the video pipeline

Place a video in data/. The following example uses data/new_video.mp4.

Use new output directory names for each run. Extraction, inference and review scripts protect their output directories from accidental reuse.

### 1. Extract frames

```bash
python -m activity_detection.prepare_video data/new_video.mp4 --out outputs/new_frames
```

The default sampling rate is two frames per second. Images are resized to fit within 640 × 640 while preserving aspect ratio.

To process only the beginning of a video, optionally add:

```text
--limit-sec 20
```

### 2. Generate initial observations

```bash
python -m activity_detection.analyze_video --manifest outputs/new_frames/manifest.json --out outputs/new_run
```

Each call uses the current frame and up to three preceding frames. The observation describes the current target timestamp.

The model loads once per process. Each completed observation is saved to observations.jsonl. Automatic resume is not implemented.

### 3. Review uncertainty and transitions

```bash
python -m activity_detection.review_video --predictions outputs/new_run/observations.json --manifest outputs/new_frames/manifest.json --out outputs/new_review
```

### 4. Generate summaries and events

```bash
python -m activity_detection.summarize_video outputs/new_review/observations.json
python -m activity_detection.summarize_bed outputs/new_review/observations.json
python -m activity_detection.video_events outputs/new_review/observations.json
```

### 5. Generate monitoring decisions

```bash
python -m activity_detection.video_monitoring --folder outputs/new_review
```

These final commands rewrite their named summary files in the
selected run directory.

## Single-image and sequence checks

```bash
python -m activity_detection.recognize_activity data/test.jpg
python -m activity_detection.analyze_sequence --manifest outputs/new_frames/manifest.json --start-sec 0
```

The sequence check requires four frames at or after the requested start time. It classifies the final selected frame.

Walking is not accepted in the single-image test.

## Context-review policy

The controller requests review for:

- Invalid structured responses.
- UNKNOWN activity or occupancy.
- Evidence quality other than clear.
- Outside-bed observations requiring movement evidence.
- Changes in activity or occupancy.

For each flagged observation:

1. Request context within approximately two seconds of the target.
2. If unresolved, expand to approximately four seconds.
3. Include at most eight frames per request.
4. Stop after at most two attempts.
5. Use UNKNOWN when review remains unresolved.

Review responses include moving_away, approaching_bed and
movement_evidence fields.

Requests, raw responses and outcomes are saved in review_log.jsonl.

The model's evidence-quality label is self-reported and is not a calibrated probability. Consistent but wrong predictions can bypass the review conditions.

## Validation and uncertainty

Validation checks allowed labels, field types and selected logical constraints. For example, LYING_IN_BED with OUT_OF_BED occupancy is rejected.

Responses with insufficient evidence or no visible person are mapped to UNKNOWN activity and occupancy.

Validation does not verify that a description matches the image, and it does not detect every contradiction between free-text evidence and structured fields.

## Timeline and duration policy

Each sampled label is held until the next observation timestamp. 
The final label is extended to the analyzed video end.

Durations are therefore estimates based on sampled observations.

Bed occupancy is summarized from bed_status observations.
UNKNOWN occupancy breaks a continuous predicted out-of-bed period.

## Bed-event rules

A meaningful exit requires:

1. A previous in-bed observation.
2. A subsequent outside-bed observation.
3. WALKING with explicit evidence of movement away from the bed.

A brief stand followed by sitting back does not confirm an exit.

A return requires:

1. Outside-bed history with approach evidence.
2. Sitting on the bed.
3. Subsequently lying down.

UNKNOWN activity breaks the event-evidence chain.

These strict rules can miss events if intermediate states are
misclassified or not captured by sampling.

## Monitoring rules

| Condition                                                           | Decision |
| Known activity and observed in bed                                  | NORMAL |
| Uncertain activity or occupancy                                     | MONITOR |
| Outside bed without a confirmed exit                                | MONITOR |
| Confirmed exit with absence below threshold                         | MONITOR |
| Confirmed exit with continuous predicted absence reaching threshold | ALERT |

The default absence threshold is 180 seconds. This is a configurable demonstration policy, not a clinically validated threshold.

```bash
python -m activity_detection.video_monitoring --folder outputs/new_review --absence-alert-sec 180
```

Unknown occupancy resets the continuous-absence timer but retains the unresolved confirmed-exit history. Observed in-bed occupancy
clears that history.

## Offline timing

The system processes recorded video and may use later frames to review an earlier timestamp.

Observation time and evidence-availability time are recorded separately. 
Additional context delay is not added to the predicted absence duration.

The implementation is not a live camera service.

## Manual annotations and evaluation

Ground-truth annotations are stored separately from predictions
in data/annotations/.

They contain:

- Video path and duration.
- Activity and occupancy intervals.
- Manually identified bed events.
- Annotation notes and timing precision.

Evaluate activities and durations:

```bash
python -m activity_detection.evaluate_video --predictions outputs/new_review/observations.json --ground-truth data/annotations/new_video_ground_truth.json
```

Evaluate events:

```bash
python -m activity_detection.evaluate_events --events outputs/new_review/bed_events.json --ground-truth data/annotations/new_video_ground_truth.json
```

### Metric definitions

- Activity accuracy: correct activity labels divided by sampled observations.
- Occupancy accuracy: correct bed-status labels divided by sampled observations.
- Confusion matrix: actual activity rows and predicted activity columns.
- Duration error: predicted duration minus annotated duration, with absolute
  error also reported for each label.
- Event precision: matched predictions divided by predicted events.
- Event recall: matched predictions divided by annotated events.

UNKNOWN labels are included in classification evaluation.

Events are matched one-to-one by event type and start time.
The default start-time tolerance is one second and is configurable with --tolerance-sec.

Metrics with undefined denominators are saved as null and displayedas N/A. 
No-event clips do not establish event-detection recall.

## Development results

| Clip and run                        | Activity accuracy | Occupancy accuracy |
| room.mp4, video_review_01           | 0/39 — 0%         | 39/39 — 100% |
| room_02.mp4, revised first pass     | 0/16 — 0%         | 10/16 — 62.5% |
| room_02.mp4, revised context review | 4/16 — 25%        | 13/16 — 81.25% |

For the second clip after review:

- Actual exits: 1.
- Predicted exits: 0.
- False negatives: 1.
- False positives: 0.
- Exit recall: 0%.
- Exit precision: N/A.
- Exit F1: 0%.

Neither clip contains an annotated return-to-bed event.

Three of the four correct activity predictions in the second reviewed clip are UNKNOWN. 
Standing and walking were still missed.

Prompts changed during development. These results are historical development runs, not a controlled benchmark of one frozen prompt.
Re-running with the current code may produce different results.

The annotations were created after inspecting model outputs, and transition times are approximate. 
An independent held-out evaluation
has not been completed.

## Tests

```bash
python -m unittest discover -s tests -v
```

The existing 19 unit tests passed:

- Seven bed-event tests.
- Seven monitoring tests.
- Five context-review tests.

Context-review tests use mocked model responses. These tests verify
selected program behaviour, not visual recognition accuracy.
The evaluation scripts do not yet have a dedicated unit-test suite.

## Output files

| File                    | Contents |
| manifest.json           | Frame filenames and sampled timestamps |
| observations.jsonl      | Incrementally saved first-pass observations |
| observations.json       | Complete prediction records |
| review_log.jsonl        | Context-review attempts and outcomes |
| activity_summary.  json | Activity intervals and duration totals |
| bed_summary.json        | Occupancy intervals and duration totals |
| bed_events.json         | Predicted bed exits and returns |
| monitoring.json         | Monitoring decisions and explanations |
| evaluation.json         | Classification and duration metrics |
| event_evaluation.json.  | Event matching and detection metrics |

## Known limitations

- Sitting can be confused with lying.
- Structured labels and descriptions can disagree.
- Consistent incorrect predictions may not trigger review.
- Earlier visual context may influence target-time predictions.
- The model may predict activity after the person leaves view.
- The evaluated bed departure was missed.
- No real-video return-to-bed sequence has been evaluated.
- No real-video prolonged-absence alert has been evaluated.
- The two clips do not establish general performance or performance
  specifically on elderly subjects.

See docs/failure_cases.md for observed failures and supporting outputs.

## Potential improvements — not implemented

- Explicit target-person and bed-region configuration.
- Better detection of inconsistent or uncertain observations.
- Comparison with alternative vision or pose-estimation approaches.
- More diverse, independently annotated test videos.
- Held-out evaluation after freezing prompts and rules.
