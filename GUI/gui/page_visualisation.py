from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QFont
from core.visualisation import generate_3d_molecule_html

class MoleculeInspectorDialog(QDialog):
    def __init__(self, smiles: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"3D Inspector: {smiles}")
        self.resize(700, 700)
        
        # Apply dark mode styling to the window
        self.setStyleSheet("""
            QDialog {
                background-color: #1A1C22;
            }
            QLabel {
                color: #d1d1d1;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title label
        title = QLabel(f"Molecular Structure: {smiles}")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        layout.addWidget(title)

        # Generate HTML and load it into the WebEngine
        self.browser = QWebEngineView()
        html_content = generate_3d_molecule_html(smiles)
        self.browser.setHtml(html_content)
        
        layout.addWidget(self.browser)