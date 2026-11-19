from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget


class CaptionOverlay(QWidget):
    """Borderless, always-on-top, click-through caption bar at the bottom of the primary screen."""

    def __init__(
        self,
        font_size=22,
        opacity=0.75,
        bottom_margin=60,
        width_ratio=0.8,
        hide_after_ms=4000,
        max_segments=2,
    ):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool  # keeps it off the taskbar
            | Qt.WindowTransparentForInput  # clicks pass through
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self._opacity = opacity
        self._bottom_margin = bottom_margin
        self._width_ratio = width_ratio
        self._max_segments = max_segments
        self._segments = []

        font = QFont("Segoe UI", font_size)
        font.setBold(True)
        self.label = QLabel("", self)
        self.label.setFont(font)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; padding: 10px 20px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._clear)
        self._hide_after_ms = hide_after_ms

        self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(0, 0, -1, -1), 14, 14)
        painter.fillPath(path, QColor(0, 0, 0, int(255 * self._opacity)))
        painter.end()

    @Slot(str)
    def set_caption(self, text):
        text = text.strip()
        if not text:
            return
        self._segments.append(text)
        self._segments = self._segments[-self._max_segments :]
        self.label.setText(" ".join(self._segments))
        self._reposition()
        self.show()
        self._hide_timer.start(self._hide_after_ms)

    def _clear(self):
        self._segments = []
        self.label.setText("")
        self.hide()

    def _reposition(self):
        screen = QApplication.primaryScreen().availableGeometry()  # excludes the taskbar
        width = int(screen.width() * self._width_ratio)
        self.setFixedWidth(width)
        self.label.setFixedWidth(width)
        self.adjustSize()
        x = screen.x() + (screen.width() - self.width()) // 2
        y = screen.y() + screen.height() - self.height() - self._bottom_margin
        self.move(x, y)
