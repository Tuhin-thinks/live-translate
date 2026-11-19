import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from overlay import CaptionOverlay
from pipeline import TranslateWorker


def main():
    app = QApplication(sys.argv)

    overlay = CaptionOverlay()
    worker = TranslateWorker(model_size="large-v3", language="ja")

    worker.text_ready.connect(overlay.set_caption)  # queued across threads
    worker.status.connect(lambda msg: print(msg, flush=True))

    def on_error(msg):
        print(f"ERROR: {msg}", file=sys.stderr, flush=True)
        app.exit(1)  # the pipeline cannot run, so stop instead of showing an empty overlay

    worker.error.connect(on_error)

    def shutdown():
        print("Shutting down...", flush=True)
        worker.stop()
        worker.wait(10000)  # lets an in-flight transcription finish

    app.aboutToQuit.connect(shutdown)

    # Ctrl+C: Python only runs signal handlers between bytecodes, and Qt's event
    # loop sits in C++. The handler asks Qt to quit; the no-op timer hands control
    # back to Python a few times a second so the handler actually gets to run.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break
        signal.signal(signal.SIGBREAK, lambda *_: app.quit())
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)

    worker.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
