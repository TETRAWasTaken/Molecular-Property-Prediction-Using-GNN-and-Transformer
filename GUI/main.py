import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QSplashScreen, QMainWindow
from gui.main_window import MainWindow
from PySide6.QtGui import QFontDatabase, QPixmap, Qt
import time

stylesheet = """
QMainWindow {
    background-color: #1A1C22;
}

QPushButton {
    background-color: #0d1661;
    color: #ffffff;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #121e87;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #0f1970;
    color: #ffffff;
}

QGroupBox {
    background-color: #2D3139;
    border: 1px solid #3A3F49;
    border-radius: 10px;
    color: #ffffff;
    margin-top: 18px;
    padding-top: 14px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: -2px;
    padding: 4px 10px;
    border-radius: 8px;
    background-color: #242830;
    color: #F2F4F8;
    letter-spacing: 0.8px;
}

QGroupBox QTextEdit,
QGroupBox QDoubleSpinBox,
QGroupBox QSpinBox,
QGroupBox QLineEdit {
    background-color: transparent;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
}

QWidget {
    font-family: 'Montserrat';
    font-size: 13px;
}

QGroupBox {
    font-family: 'Montserrat';
    font-weight: bold;
    font-size: 16px;
}

QLabel {
    color: #d1d1d1;
}

QPushButton#runButton {
    font-family: 'Montserrat';
    font-weight: bold;
    font-size: 16px;
    letter-spacing: 1px;
}

QTextEdit {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
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