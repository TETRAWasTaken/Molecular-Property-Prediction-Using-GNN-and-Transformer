import sys
import os
from pathlib import Path

existing_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
if "--disable-skia-graphite" not in existing_flags:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        f"{existing_flags} --disable-skia-graphite".strip()
    )

from PySide6.QtWidgets import QApplication, QSplashScreen, QMainWindow
from gui.main_window import MainWindow
from PySide6.QtGui import QFontDatabase, QPixmap, Qt
from PySide6.QtCore import QThread
import time
from core.inference import InferenceThread

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
    
    pixmap = QPixmap("/Users/anshumaansoni/Downloads/ACTION.png")
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()

    app.processEvents()  # Ensure the splash screen is displayed immediately
    time.sleep(2)
    
    app.setStyle("Fusion") 
    
    window = MainWindow()
    window.show()
    
    splash.finish(window)
    sys.exit(app.exec())