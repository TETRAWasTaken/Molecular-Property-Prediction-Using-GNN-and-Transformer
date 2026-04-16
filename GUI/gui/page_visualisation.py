from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QFont
from core.visualisation import generate_3d_molecule_html_file

class MoleculeInspectorDialog(QDialog):
    def __init__(self, smiles: str, explainability=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"3D Inspector: {smiles}")
        self.resize(700, 500)
        self.smiles = smiles
        self.explainability = explainability or {}
        self._last_html_path = None
        
        self.setStyleSheet("""
            QDialog {
                background-color: #FAF9F6;
            }
            QLabel {
                color: #4A3A2A;
            }
            QCheckBox {
                color: #7A6657;
                font-size: 11px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Title label
        title = QLabel(f"Molecular Structure: {smiles}")
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(title)

        self.attention_toggle = QCheckBox("Attention mode (highlight transformer-focused bonds)")
        self.attention_toggle.toggled.connect(self._refresh_html)
        layout.addWidget(self.attention_toggle)

        # Generate HTML and load it into the WebEngine
        self.browser = QWebEngineView()
        self._refresh_html()
        layout.addWidget(self.browser, stretch=2)

    def _refresh_html(self):
        html_path = generate_3d_molecule_html_file(
            self.smiles,
            atom_contributions=self.explainability.get("atom_scores"),
            attention_bonds=self.explainability.get("bond_scores"),
            attention_mode=self.attention_toggle.isChecked(),
        )
        self._cleanup_last_html()
        self._last_html_path = html_path
        self.browser.load(QUrl.fromLocalFile(html_path))

    def _cleanup_last_html(self):
        if not self._last_html_path:
            return
        try:
            Path(self._last_html_path).unlink(missing_ok=True)
        except Exception:
            pass

    def closeEvent(self, event):
        self._cleanup_last_html()
        super().closeEvent(event)