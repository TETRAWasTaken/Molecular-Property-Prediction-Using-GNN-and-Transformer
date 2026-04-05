import sys
import os
from pathlib import Path

existing_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
if "--disable-skia-graphite" not in existing_flags:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        f"{existing_flags} --disable-skia-graphite".strip()
    )

from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from gui.main_window import MainWindow
from PySide6.QtGui import QFontDatabase, Qt
from PySide6.QtCore import QThread, QTimer, QUrl
from core.inference import EngineWarmupThread

stylesheet = """
/* Main Application */
QMainWindow {
    background-color: #1A1C22;
}

QWidget {
    font-family: 'Montserrat', sans-serif;
    font-size: 13px;
}

/* Labels */
QLabel {
    color: #d1d1d1;
}

/* Group Boxes */
QGroupBox {
    background-color: #252a32;
    border: 2px solid #3A3F49;
    border-radius: 12px;
    color: #ffffff;
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
    background-color: #242830;
    color: #F2F4F8;
    letter-spacing: 0.8px;
}

/* Input Fields */
QTextEdit {
    background-color: #1f2329;
    border: 2px solid #3A3F49;
    border-radius: 8px;
    color: #e0e0e0;
    padding: 8px;
    font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
    font-size: 13px;
}

QTextEdit:focus {
    border: 2px solid #5B6EFF;
    background-color: #222830;
}

QDoubleSpinBox, QSpinBox, QLineEdit {
    background-color: #1f2329;
    border: 2px solid #3A3F49;
    border-radius: 6px;
    color: #e0e0e0;
    padding: 6px 8px;
    font-size: 12px;
}

QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
    border: 2px solid #5B6EFF;
    background-color: #222830;
}

/* Buttons */
QPushButton {
    background-color: #0d1661;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.5px;
}

QPushButton:hover {
    background-color: #1521a0;
}

QPushButton:pressed {
    background-color: #0f1970;
}

QPushButton#runButton {
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 1px;
    padding: 12px 24px;
}

/* Tables */
QTableWidget {
    background-color: #1f2329;
    alternate-background-color: #252a32;
    gridline-color: #3A3F49;
    border: 1px solid #3A3F49;
    border-radius: 8px;
}

QTableWidget::item {
    padding: 8px;
    border-bottom: 1px solid #2d3238;
}

QTableWidget::item:selected {
    background-color: #1521a0;
    color: #ffffff;
}

QTableWidget::item:hover {
    background-color: #2d3238;
}

QHeaderView::section {
    background-color: #0d1661;
    color: #ffffff;
    padding: 10px;
    border: none;
    border-bottom: 2px solid #5B6EFF;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QTableWidget:focus {
    border: 2px solid #5B6EFF;
}

/* Scroll Bars */
QScrollBar:vertical {
    background-color: #1f2329;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #5B6EFF;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #7a8dff;
}

QScrollBar:horizontal {
    background-color: #1f2329;
    height: 10px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #5B6EFF;
    border-radius: 5px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #7a8dff;
}
"""

class VideoSplashScreen(QDialog):
    """A video splash screen that plays a video file during app startup."""
    def __init__(self, video_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setModal(False)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.resize(960, 540)
        
        # Center on screen
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
        self.audio.setVolume(0.0)  # Mute splash video
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)
        self.player.setSource(QUrl.fromLocalFile(str(video_path)))
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)

    def _on_media_status_changed(self, status):
        """Auto-loop video or handle end of playback."""
        if status == QMediaPlayer.EndOfMedia:
            self.player.setPosition(0)
            self.player.play()

    def start(self):
        """Show the splash and start video playback."""
        self.show()
        self.player.play()

    def stop(self):
        """Stop video and close splash."""
        self.player.stop()
        self.close()

if __name__ == "__main__":

    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)

    font_path = Path(__file__).resolve().parent / "assets" / "Montserrat-VariableFont_wght.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    if font_id == -1:
        print(f"[Font] Failed to load: {font_path}")
    else:
        families = QFontDatabase.applicationFontFamilies(font_id)
        print(f"[Font] Loaded: {font_path} -> {families}")
    
    # Load video splash screen
    video_splash_path = Path("/Users/anshumaansoni/Downloads/Splash.mp4")
    video_splash = None
    
    if video_splash_path.exists():
        video_splash = VideoSplashScreen(video_splash_path)
        video_splash.start()
        print(f"[Splash] Video loaded: {video_splash_path}")
    else:
        print(f"[Splash] Video not found: {video_splash_path}")

    # Warm up ONNX Runtime engine in parallel while the app is launching.
    engine_warmup_thread = EngineWarmupThread()

    def _on_engine_ready(model_path):
        print(f"[Engine] Warm-up complete: {model_path}")

    def _on_engine_error(message):
        print(f"[Engine] Warm-up failed: {message}")

    engine_warmup_thread.ready.connect(_on_engine_ready)
    engine_warmup_thread.error.connect(_on_engine_error)
    engine_warmup_thread.start()

    app.processEvents()  # Ensure the splash screen is displayed immediately

    app.setStyle("Fusion")

    window = MainWindow()
    window.resize(1920, 1080)
    window.setWindowTitle("Molecular Property Prediction and Recommendation")
    window.engine_warmup_thread = engine_warmup_thread

    def _show_main_window():
        window.show()
        if video_splash is not None:
            video_splash.stop()

    # Do not block with sleep; let event loop paint splash, then open main window.
    QTimer.singleShot(8000, _show_main_window)
    sys.exit(app.exec())