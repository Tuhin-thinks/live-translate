# live-translate

**Real-time AI subtitles for anything playing on your Windows PC.** Watch Japanese anime, streams, games, or videos with live English captions — translated offline by Whisper, on your own GPU.

![Platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-0078D6?logo=windows)
![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/model-faster--whisper%20large--v3-412991)
![CUDA](https://img.shields.io/badge/GPU-CUDA%20accelerated-76B900?logo=nvidia&logoColor=white)
![Offline](https://img.shields.io/badge/runs-100%25%20offline-success)

live-translate listens to your system audio, so it works with a YouTube video, an anime episode, a game, a stream or a call. It translates the speech with a local Whisper model and shows the result as a borderless caption bar at the bottom of your screen. Everything runs on your machine, with no cloud service and no API key.

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│                 (your video / game / call)                   │
│                                                              │
│      ╭──────────────────────────────────────────────╮        │
│      │  I've never seen anything like this before.  │        │
│      ╰──────────────────────────────────────────────╯        │
└──────────────────────────────────────────────────────────────┘
```

## Features

- **Any audio source.** Captures your speakers or headphones through WASAPI loopback, so it doesn't care which app is playing.
- **Local and private.** Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) `large-v3`, and the audio never leaves your PC.
- **Caption overlay.** Always on top and click-through, and it stays off the taskbar. It sits above the taskbar and hides itself after a few seconds of quiet.
- **Splits at natural breaks.** Audio is cut into phrases at pauses, and where speech gives way to music or sound effects, rather than on a fixed timer. Words are not chopped in half, and music on its own is never sent for translation.
- **Filters made-up lines.** Whisper sometimes invents lines like *"Thanks for watching!"* over music; these are filtered out.
- **Works with or without a GPU.** It uses CUDA if an NVIDIA GPU is available and falls back to the CPU otherwise.
- **Keeps up.** If translation falls behind, stale phrases are skipped so the captions stay close to live.

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11 (loopback capture uses WASAPI) |
| Python | 3.12+, managed by [uv](https://docs.astral.sh/uv/) |
| GPU (recommended) | NVIDIA GPU with about 4 GB of VRAM, a recent driver, and the [CUDA 12 toolkit](https://developer.nvidia.com/cuda-downloads) (for cuBLAS) |
| Disk | About 3.5 GB for the `large-v3` model |

On the CPU it works but is slow. `large-v3` takes several seconds per phrase, so expect skipped captions.

## Quick start

```powershell
# 1. Install uv (once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Get the code and install dependencies
git clone https://github.com/Tuhin-thinks/live-translate.git
cd live-translate
uv sync

# 3. Run
uv run main.py
```

The first run downloads the model into `./models/` (about 3 GB). After that, startup takes a few seconds.

Start playing something in Japanese and captions will appear at the bottom of the screen. Press **Ctrl+C** in the terminal to quit.

### What the console shows

```
Loading model 'large-v3' on cuda (int8_float16, cache: ...\models)...
Listening...
Capturing: Headphones (Realtek(R) Audio)
[2.8s audio -> 0.61s] That I've never seen before
  dropped 'Thank you for watching!': stock phrase
```

Each line shows how much audio was translated, how long translation took, and the English text. Lines starting with `dropped` are made-up lines the filter caught.

## Usage tips

- **Source language.** Japanese is the default. To change it, edit `language="ja"` in [`main.py`](main.py) and use any [Whisper language code](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py), such as `ko`, `zh`, `es` or `de`. The output is always English, because that is the only language Whisper's `translate` task produces.
- **Output device.** It captures the *default* playback device. Switch the Windows default output to choose which device is captioned.
- **Fullscreen.** Use **borderless or windowed fullscreen** in games and players. Exclusive fullscreen can draw over the overlay.
- **A smaller or faster model.** Pass `model_size="medium"` or `"small"` to `TranslateWorker` in `main.py`. These are faster and use less memory, but translate less accurately.

You'll find more knobs in [docs/configuration.md](docs/configuration.md).

## How it works

```
system audio ──► capture thread ──► phrase segmenter ──► Whisper (translate) ──► filter ──► caption overlay
 (loopback)       32 ms blocks       VAD + loudness        faster-whisper         drops      PySide6
```

1. **Capture.** `soundcard` records the loopback of the default speaker at 16 kHz mono, in 32 ms blocks, on its own thread.
2. **Segment.** Each block gets a speech probability from a streaming Silero VAD and a loudness value. A phrase ends after about 250 ms of silence or non-speech, and is force-cut at 6 s.
3. **Translate.** faster-whisper runs `task="translate"` on each phrase.
4. **Filter.** Segments that look made up are dropped.
5. **Display.** The text is sent to the Qt overlay through a signal that is safe to use across threads.

The full walkthrough is in [docs/architecture.md](docs/architecture.md).

## Project layout

| File | Purpose |
|---|---|
| [`main.py`](main.py) | Entry point. Wires the worker to the overlay and handles Ctrl+C. |
| [`pipeline.py`](pipeline.py) | `TranslateWorker`: capture, translation and the made-up-line filter |
| [`segmenter.py`](segmenter.py) | `PhraseSegmenter`: decides where to cut the audio stream |
| [`speech.py`](speech.py) | `SileroStream`: streaming voice activity detection |
| [`overlay.py`](overlay.py) | `CaptionOverlay`: the borderless caption window |
| [`download_model.py`](download_model.py) | Optional standalone model downloader with logging |
| [`docs/`](docs/) | Architecture, configuration and troubleshooting |

## Troubleshooting

The usual problems are:

- the GPU not being detected (laptops in Eco or iGPU mode),
- `cublas64_12.dll` not being found,
- the caption overlay not showing,
- made-up captions.

See [docs/troubleshooting.md](docs/troubleshooting.md) for fixes.

## Limitations

- Windows only, because capture relies on WASAPI loopback.
- Output is English only.
- Captions lag the audio by about the length of a phrase plus translation time, typically 1–4 s on a GPU.
- Overlapping speakers and loud background music lower accuracy.

## Acknowledgements

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [CTranslate2](https://github.com/OpenNMT/CTranslate2)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [SoundCard](https://github.com/bastibe/SoundCard)
- [PySide6 / Qt](https://doc.qt.io/qtforpython-6/)
