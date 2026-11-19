"""Streaming speech detection with the Silero VAD model bundled in faster-whisper."""

import os

import numpy as np


class SileroStream:
    """Returns a speech probability for each consecutive 512-sample window (32 ms at 16 kHz).

    faster-whisper's own ``SileroVADModel`` resets its recurrent state on every call,
    which suits whole files but not a live stream. This keeps the LSTM state and the
    64-sample context between calls, so each window is judged with what came before it.
    """

    WINDOW = 512
    CONTEXT = 64

    def __init__(self):
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._session = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), "silero_vad_v6.onnx"),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.reset()

    def reset(self):
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def __call__(self, window):
        window = np.asarray(window, dtype=np.float32).reshape(1, -1)
        if window.shape[1] != self.WINDOW:
            raise ValueError(f"expected {self.WINDOW} samples, got {window.shape[1]}")
        x = np.concatenate([self._context, window], axis=1)
        prob, self._h, self._c = self._session.run(
            None, {"input": x, "h": self._h, "c": self._c}
        )
        self._context = window[:, -self.CONTEXT :]
        return float(prob.reshape(-1)[0])
