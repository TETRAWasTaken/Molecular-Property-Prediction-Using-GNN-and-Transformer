"""Launch the PySide6 application with a styled main window and optional splash video.

The module loads the application stylesheet, optionally plays a splash screen,
warms up the inference engine in the background, and opens the main GUI window
after startup.
"""

import sys
import os
from pathlib import Path

existing_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
# Mitigate QtWebEngine Chromium Skia mailbox errors on macOS while keeping WebGL on.
flags_to_add = [
    "--disable-skia-graphite",
    "--disable-features=UseSkiaRenderer",
]
if os.environ.get("GUI_FORCE_SOFTWARE_WEBENGINE", "0") == "1":
    flags_to_add.extend(["--disable-gpu", "--disable-gpu-compositing"])
updated_flags = existing_flags
for flag in flags_to_add:
    if flag not in updated_flags:
        updated_flags = f"{updated_flags} {flag}".strip()
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = updated_flags

from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from gui.main_window import MainWindow
from PySide6.QtGui import QFontDatabase, Qt
from PySide6.QtCore import QTimer, QUrl
from core.inference import EngineWarmupThread

stylesheet = """
QMainWindow {
    background-color: #FAF9F6;
}

QWidget {
    font-family: 'Montserrat', sans-serif;
    font-size: 13px;
}

QLabel {
    color: #4A3A2A;
}

QGroupBox {
    background-color: #FFFDF8;
    border: 2px solid #E0C7B1;
    border-radius: 12px;
    color: #4A3A2A;
    margin-top: 18px;
    padding-top: 14px;
    padding-left: 12px;
    padding-right: 12px;
    padding-bottom: 12px;
    font-family: 'Montserrat';
    font-weight: 600;
    font-size: 15px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: -2px;
    padding: 6px 12px;
    border-radius: 6px;
    background-color: #CC5500;
    color: #FAF9F6;
    letter-spacing: 0.8px;
}

QTextEdit {
    background-color: #FFF8F2;
    border: 2px solid #E0C7B1;
    border-radius: 8px;
    color: #4A3A2A;
    padding: 8px;
    font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
    font-size: 13px;
}

QTextEdit:focus {
    border: 2px solid #CC5500;
    background-color: #FFFDF8;
}

QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #FFF8F2;
    border: 2px solid #E0C7B1;
    border-radius: 6px;
    color: #4A3A2A;
    padding: 6px 8px;
    font-size: 12px;
}

QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
    border: 2px solid #CC5500;
    background-color: #FFFDF8;
}

QPushButton {
    background-color: #CC5500;
    color: #FAF9F6;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.5px;
}

QPushButton:hover {
    background-color: #B34700;
}

QPushButton:pressed {
    background-color: #8F3B00;
}

QPushButton#runButton {
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 1px;
    padding: 12px 24px;
}

QTableWidget {
    background-color: #FFFDF8;
    alternate-background-color: #F6EEE4;
    gridline-color: #E2CDBC;
    border: 1px solid #E0C7B1;
    border-radius: 8px;
}

QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #E8D8CA;
}

QTableWidget::item:selected {
    background-color: #F1D1B9;
    color: #4A3A2A;
}

QTableWidget::item:hover {
    background-color: #F7E5D8;
}

QHeaderView::section {
    background-color: #CC5500;
    color: #FAF9F6;
    padding: 10px;
    border: none;
    border-bottom: 2px solid #B34700;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QTableWidget:focus {
    border: 2px solid #CC5500;
}

QScrollBar:vertical {
    background-color: #F3E8DE;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #CC5500;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #B34700;
}

QScrollBar:horizontal {
    background-color: #F3E8DE;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #CC5500;
    border-radius: 5px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #B34700;
}
"""

class VideoSplashScreen(QDialog):
    """A video splash screen that plays a video file during app startup.

    Args:
        video_path: Absolute or relative path to the splash video file.
        parent: Optional parent widget.
    """

    def __init__(self, video_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setModal(False)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.resize(960, 540)

        screen = QApplication.primaryScreen()
        geo = self.frameGeometry()
        center = screen.availableGeometry().center()
        geo.moveCenter(center)
        self.move(geo.topLeft())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.0)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)
        self.player.setSource(QUrl.fromLocalFile(str(video_path)))
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)

    def _on_media_status_changed(self, status):
        """Auto-loop the video when playback reaches the end.

        Args:
            status: Current media status emitted by the player.
        """

        if status == QMediaPlayer.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()

    def start(self):
        """Show the splash screen and begin playback.

        Args:
            None.
        """

        self.show()
        self.player.play()

    def stop(self):
        """Stop playback and close the splash screen.

        Args:
            None.
        """

        self.player.stop()
        self.close()


def main():
    """Start the GUI application, show the splash screen, and open the main window.

    Args:
        None.
    """

    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)

    font_path = Path(__file__).resolve().parent / "assets" / "Montserrat-VariableFont_wght.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    if font_id == -1:
        print(f"[Font] Failed to load: {font_path}")
    else:
        families = QFontDatabase.applicationFontFamilies(font_id)
        print(f"[Font] Loaded: {font_path} -> {families}")

    video_splash_path = Path(r"GUI/assets/splash.mp4")
    video_splash = None

    if video_splash_path.exists():
        video_splash = VideoSplashScreen(video_splash_path)
        video_splash.start()
        print(f"[Splash] Video loaded: {video_splash_path}")
    else:
        print(f"[Splash] Video not found: {video_splash_path}")

    def _on_engine_ready(model_path):
        """Report successful warm-up completion.

        Args:
            model_path: Path to the warmed-up model artifact.
        """

        print(f"[Engine] Warm-up complete: {model_path}")

    def _on_engine_error(message):
        """Report warm-up failures to the console.

        Args:
            message: Error message emitted by the warm-up worker.
        """

        print(f"[Engine] Warm-up failed: {message}")

    app.processEvents()
    app.setStyle("Fusion")

    window = MainWindow()
    window.resize(1920, 1080)
    window.setWindowTitle("Molecular Property Prediction and Recommendation")

    def _cleanup_on_quit():
        """Drain background tasks before Qt tears down widgets."""

        window.cleanup_background_tasks()
        if video_splash is not None:
            video_splash.stop()

    app.aboutToQuit.connect(_cleanup_on_quit)

    def _start_engine_warmup():
        """Start model warm-up after the UI is already visible."""

        engine_warmup_thread = EngineWarmupThread()
        engine_warmup_thread.ready.connect(_on_engine_ready)
        engine_warmup_thread.error.connect(_on_engine_error)
        window.engine_warmup_thread = engine_warmup_thread
        engine_warmup_thread.start()

    def _show_main_window():
        """Display the main window and stop the splash screen if it is running.

        Args:
            None.
        """

        window.show()
        if video_splash is not None:
            video_splash.stop()
        QTimer.singleShot(0, _start_engine_warmup)

    QTimer.singleShot(8000, _show_main_window)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()