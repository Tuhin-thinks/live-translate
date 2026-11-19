# Configuration

All settings are constructor arguments; there is no config file yet. Change them in [`main.py`](../main.py), where the worker and the overlay are created.

## `TranslateWorker` (pipeline.py)

```python
worker = TranslateWorker(model_size="large-v3", language="ja")
```

| Argument | Default | Notes |
|---|---|---|
| `model_size` | `"large-v3"` | Any faster-whisper model name (`small`, `medium`, `large-v2`, `large-v3`, …) or a local CTranslate2 model folder. Smaller models are faster and less accurate. `turbo` models **cannot translate**. |
| `language` | `"ja"` | Source language code. Set it to `None` to auto-detect, which is slower and less reliable on short phrases. |
| `device` | auto | `"cuda"` or `"cpu"`. Auto-detect picks CUDA when a GPU is visible. |
| `compute_type` | auto | `int8_float16` on CUDA, `int8` on CPU. Try `float16` on a GPU with plenty of VRAM. |
| `min_seconds` | `1.0` | Shortest phrase cut at a pause. Lower gives snappier captions but less context. |
| `max_seconds` | `6.0` | Longest phrase before a forced cut. |
| `pause_ms` | `250` | How long a pause or non-speech stretch must last to end a phrase. Raise it if sentences are being split mid-thought. |
| `max_pending` | `2` | Phrases queued for translation before the oldest is dropped. |

Models download to `./models/` (see `MODELS_DIR` in `pipeline.py`).

### Rough model guide (NVIDIA GPU)

| Model | VRAM (int8_float16) | Speed | Translation quality |
|---|---|---|---|
| `small` | ~1 GB | very fast | rough |
| `medium` | ~2 GB | fast | decent |
| `large-v3` | ~3–4 GB | ~0.5–3 s per phrase | best |

## `PhraseSegmenter` (segmenter.py)

These are tuned for typical video audio. You only need to change them if the splitting misbehaves.

| Argument | Default | Effect |
|---|---|---|
| `speech_on` | `0.5` | VAD probability at which a block counts as confident speech |
| `speech_off` | `0.35` | VAD probability below which a block counts as a gap |
| `rel_threshold` | `0.3` | A block is "quiet" below this fraction of the recent loudness peak |
| `abs_floor` | `0.003` | Absolute RMS below which a block is always quiet |
| `level_decay` | `0.995` | How fast the loudness peak decays, per 32 ms block |
| `min_speech_ms` | `200` | Chunks with less speech than this are discarded |
| `preroll_ms` | `200` | Audio kept before the first speech, so the onset isn't clipped |

## Hallucination filter (pipeline.py)

| Constant | Default | Effect |
|---|---|---|
| `NO_SPEECH_SOFT` | `0.5` | With `LOW_LOGPROB`, a segment is dropped when both thresholds are crossed |
| `LOW_LOGPROB` | `-1.0` | Average token log-probability below which a decode counts as low-confidence |
| `MAX_COMPRESSION_RATIO` | `2.4` | Drops repetitive or looping output |
| `HALLUCINATION_PHRASES` | set | Exact stock phrases to drop, matched after normalising |

If real lines are being dropped, look at the `dropped ...` log lines to see which rule fired.

## `CaptionOverlay` (overlay.py)

```python
overlay = CaptionOverlay(font_size=26, opacity=0.6, bottom_margin=100)
```

| Argument | Default | Effect |
|---|---|---|
| `font_size` | `22` | Caption font size (Segoe UI, bold) |
| `opacity` | `0.75` | Opacity of the background box (0 to 1) |
| `bottom_margin` | `60` | Pixels above the taskbar |
| `width_ratio` | `0.8` | Box width as a fraction of the screen width |
| `hide_after_ms` | `4000` | Hide after this long without new text |
| `max_segments` | `2` | Number of recent phrases shown together |
