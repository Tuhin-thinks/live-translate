# Troubleshooting

## `CUDA failed with error no CUDA-capable device is detected`

The app falls back to the CPU only if it sees no GPU at startup. This error means ctranslate2 found a GPU entry, but the GPU is not usable.

- **Gaming laptops (ASUS, Lenovo, MSI…):** the dGPU may be switched off by an *Eco*, *iGPU-only* or *Hybrid-off* GPU mode. Power mode is a separate setting. In Armoury Crate (or your vendor's tool), set **GPU Mode** to *Standard* or *Ultimate*, plug in AC power, and reboot if asked.
- Check that the GPU is visible with `nvidia-smi`. If it isn't listed, Windows can't see it either.
- Check what ctranslate2 sees:
  ```powershell
  uv run python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"
  ```

## `Library cublas64_12.dll is not found or cannot be loaded`

Install the [CUDA 12 toolkit](https://developer.nvidia.com/cuda-downloads) and make sure `CUDA_PATH` points to it. `pipeline.py` adds `%CUDA_PATH%\bin` to the DLL search path automatically. You don't need to install cuDNN separately.

## Captions are slow, or `Falling behind: dropped a stale phrase`

Whisper is slower than real time on your hardware.

- Make sure the log says `on cuda`, not `on cpu`.
- On laptops, plug in AC power, because the GPU is throttled on battery.
- Use a smaller model (`model_size="medium"` or `"small"`).

## No captions appear

- Look at the console. `[x.xs audio -> …] (nothing)` means Whisper heard no speech, or the filter dropped everything (check the `dropped` lines).
- If `Capturing: …` shows the wrong device, change the Windows default output device and restart.
- Audio sent to a different device than the default (for example, an app set to a specific output) is not captured.
- Exclusive fullscreen apps can cover the overlay. Use borderless or windowed fullscreen.

## Real lines show up as `dropped …`

The log line names the rule that fired. See [configuration.md](configuration.md#hallucination-filter-pipelinepy) to loosen that rule.

## Made-up captions over music ("Thanks for watching!", "Please subscribe")

These are caught by the stock-phrase list. If a new one shows up, add its lowercased text with no punctuation to `HALLUCINATION_PHRASES` in `pipeline.py`.

## `SoundcardRuntimeWarning: data discontinuity in recording`

This is harmless; the audio driver briefly skipped a bit of sound. The app hides this warning. If you see it, `soundcard` was probably imported somewhere before the filter was installed.

## `QWindowsContext: OleInitialize() failed: COM error 0x80010106`

Something imported `soundcard` on the GUI thread. It must only be imported inside the capture thread. See [architecture.md](architecture.md#1-capture-pipelinepy-_capture_loop).

## Huggingface symlink warning on first download

This is harmless on Windows without Developer Mode; files are copied instead of symlinked.
