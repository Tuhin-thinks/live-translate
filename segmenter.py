"""Split a live audio stream into short phrases at pauses and at non-speech sounds."""

import numpy as np


class PhraseSegmenter:
    """Accumulates fixed-size audio blocks and emits phrase chunks for translation.

    Each block gets two scores: loudness (RMS) and a speech probability from a
    voice-activity model. A block is a **gap** when it is quiet *or* not speech,
    so phrases end both at silent pauses and where speech gives way to music or
    sound effects that stay loud.

    - A chunk is cut once at least ``min_seconds`` is buffered and the stream has
      been in a gap for ``pause_ms``. The cut lands on the quietest block of that
      gap, a local minimum of loudness.
    - If speech runs past ``max_seconds`` with no gap, the cut is forced at the
      least speech-like, quietest point of the buffer's second half; the rest
      carries over into the next chunk so no audio is lost.
    - Audio before any speech is dropped (keeping a short pre-roll), and a chunk
      with less than ``min_speech_ms`` of speech is never emitted, so music, or a
      stray speech-like blip at its edge, never reaches Whisper (where it would
      invite hallucinations like "Thanks for watching!").

    "Quiet" is relative: RMS below ``rel_threshold`` x the recent loudness peak
    (which decays slowly), so it adapts to playback volume.
    """

    def __init__(
        self,
        vad,
        sample_rate=16000,
        block_size=512,
        min_seconds=1.0,
        max_seconds=6.0,
        pause_ms=250,
        min_speech_ms=200,
        speech_on=0.5,
        speech_off=0.35,
        rel_threshold=0.3,
        abs_floor=0.003,
        level_decay=0.995,
        smoothing_blocks=3,
        preroll_ms=200,
    ):
        self.vad = vad  # callable: block -> speech probability in [0, 1]
        self.block_size = block_size
        block_ms = 1000 * block_size / sample_rate
        self.min_blocks = round(min_seconds * 1000 / block_ms)
        self.max_blocks = round(max_seconds * 1000 / block_ms)
        self.pause_blocks = max(1, round(pause_ms / block_ms))
        self.preroll_blocks = max(1, round(preroll_ms / block_ms))
        self.min_speech_blocks = max(1, round(min_speech_ms / block_ms))
        self.speech_on = speech_on
        self.speech_off = speech_off
        self.rel_threshold = rel_threshold
        self.abs_floor = abs_floor
        self.level_decay = level_decay
        self.smoothing_blocks = smoothing_blocks

        self._blocks = []
        self._rms = []
        self._prob = []
        self._level = 0.0

    @property
    def quiet_threshold(self):
        return max(self.abs_floor, self.rel_threshold * self._level)

    def _is_speech(self, i):
        return self._prob[i] >= self.speech_on and self._rms[i] >= self.abs_floor

    def _is_gap(self, i):
        return self._rms[i] < self.quiet_threshold or self._prob[i] < self.speech_off

    def push(self, block):
        """Add one block of mono float32 audio; return a list of finished chunks."""
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        self._level = max(rms, self._level * self.level_decay)
        self._blocks.append(block)
        self._rms.append(rms)
        self._prob.append(self.vad(block))

        n = len(self._blocks)

        # No speech yet: drop what came before, keeping a short pre-roll so the
        # first syllable is not clipped.
        if not any(self._is_speech(i) for i in range(n)):
            if n > self.preroll_blocks:
                self._drop(n - self.preroll_blocks)
            return []

        # Natural boundary: a long-enough gap (silence or non-speech) after the minimum length.
        if n >= self.min_blocks:
            gap_run = 0
            for i in range(n - 1, -1, -1):
                if not self._is_gap(i):
                    break
                gap_run += 1
            if gap_run >= self.pause_blocks:
                start = n - gap_run
                cut = start + int(np.argmin(self._rms[start:]))
                return self._emit(cut + 1)

        # No boundary in sight: force a cut at the least speech-like, quietest dip.
        if n >= self.max_blocks:
            score = self._cut_score()
            lo = max(self.min_blocks, n // 2)
            cut = lo + int(np.argmin(score[lo:n]))
            return self._emit(cut + 1)

        return []

    def flush(self):
        """Emit whatever is buffered (e.g. on shutdown)."""
        return self._emit(len(self._blocks))

    def _cut_score(self):
        """Lower is a better place to cut: low speech probability and low loudness."""
        prob = np.asarray(self._prob)
        loud = np.asarray(self._rms) / max(self._level, 1e-9)
        score = prob + loud
        k = self.smoothing_blocks
        if k > 1 and len(score) >= k:
            score = np.convolve(score, np.ones(k) / k, mode="same")
        return score

    def _drop(self, count):
        del self._blocks[:count], self._rms[:count], self._prob[:count]

    def _emit(self, end):
        speech_blocks = sum(self._is_speech(i) for i in range(end))
        blocks = self._blocks[:end]
        self._drop(end)
        if not blocks or speech_blocks < self.min_speech_blocks:
            return []
        return [np.concatenate(blocks)]
