from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QScrollArea, QLabel, QTextEdit, QPushButton, QFileDialog,
    QGraphicsDropShadowEffect, QFrame
)
from PySide6.QtGui import QColor, QFont
from PySide6.QtCore import Signal
from gui.effects import Shadow, Glow

class InputPage(QWidget):
    """
    This class represents the input page of the GUI, where users can input parameters for the simulation.
    """

    run_screening_signal = Signal(object)


    def __init__(self):
        super().__init__()

        self.properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                           'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        self.default_property_ranges = {
            'mu': (-1.686439, 3.281758),
            'alpha': (-2.641577, 2.431582),
            'homo': (-2.951540, 5.050764),
            'lumo': (-2.430256, 3.764272),
            'gap': (-1.959863, 2.107400),
            'r2': (-1.829311, 3.721191),
            'zpve': (-2.553265, 2.277082),
            'u0': (-3.888944, 2.122974),
            'u298': (-3.884573, 2.118524),
            'h298': (-3.886974, 2.121083),
            'g298': (-3.888423, 2.122296),
            'cv': (-2.001001, 2.176433),
        }
        self.uploaded_csv_path = None
    
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Keep distinct effect instances alive per widget.
        self.left_shadow = Shadow(color=QColor(255, 255, 255, 20),
                                  blur_radius=20,
                                  x_offset=0,
                                  y_offset=0).effect
        self.right_shadow = Shadow(color=QColor(255, 255, 255, 20),
                                   blur_radius=20,
                                   x_offset=0,
                                   y_offset=0).effect
        self.upload_glow = Glow(color=QColor(91, 110, 255, 80),
                                blur_radius=15,
                                x_offset=0,
                                y_offset=0).glow
        self.run_glow = Glow(color=QColor(91, 110, 255, 120),
                             blur_radius=20,
                             x_offset=0,
                             y_offset=0).glow

        ########## Column 1 : Property Ranges ##########
        left_group = QGroupBox("Define Target Property Ranges")
        left_group.setGraphicsEffect(self.left_shadow)

        left_layout = QVBoxLayout(left_group)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(12, 12, 12, 12)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
        """)
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: transparent;")
        form_layout = QFormLayout(scroll_content)
        form_layout.setSpacing(12)
        form_layout.setContentsMargins(0, 0, 0, 0)

        self.spinboxes = {}
        for prop in self.properties:
            min_box = QDoubleSpinBox()
            max_box = QDoubleSpinBox()
            min_box.setRange(-1000, 1000)
            max_box.setRange(-1000, 1000)
            default_min, default_max = self.default_property_ranges.get(prop, (100, 100))
            min_box.setValue(default_min)
            max_box.setValue(default_max)
            min_box.setMinimumHeight(32)
            max_box.setMinimumHeight(32)

            row_layout = QHBoxLayout()
            row_layout.setSpacing(8)
            row_layout.addWidget(min_box, 1)
            row_layout.addWidget(QLabel(" - "), 0)
            row_layout.addWidget(max_box, 1)

            label = QLabel(f"{prop.upper()}:")
            label_font = QFont()
            label_font.setWeight(QFont.Weight.DemiBold)
            label.setFont(label_font)
            form_layout.addRow(label, row_layout)
            self.spinboxes[prop] = {'min': min_box, 'max': max_box}

        scroll_area.setWidget(scroll_content)
        left_layout.addWidget(scroll_area)
        main_layout.addWidget(left_group, stretch=1)

        ########## Column 2 : Smiles Inputs ##########
        # Manual input of SMILES strings
        right_group = QGroupBox("Input SMILES Strings")
        right_group.setGraphicsEffect(self.right_shadow)
        right_layout = QVBoxLayout(right_group)
        right_layout.setSpacing(12)
        right_layout.setContentsMargins(12, 12, 12, 12)

        input_label = QLabel("Enter COMMA-SEPARATED SMILES:")
        input_label_font = QFont()
        input_label_font.setWeight(QFont.Weight.DemiBold)
        input_label.setFont(input_label_font)
        right_layout.addWidget(input_label)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("e.g. CCO, c1ccccc1, CC(=O)O")
        self.text_input.setMinimumHeight(100)
        right_layout.addWidget(self.text_input)

        separator_label = QLabel("- OR -")
        separator_label.setStyleSheet("color: #767676; text-align: center; margin: 8px 0px;")
        separator_label_font = QFont()
        separator_label_font.setWeight(QFont.Weight.DemiBold)
        separator_label.setFont(separator_label_font)
        right_layout.addWidget(separator_label)

        # File upload for SMILES strings
        self.btn_upload = QPushButton("📁 Upload CSV File")
        self.btn_upload.setGraphicsEffect(self.upload_glow)
        self.btn_upload.setMinimumHeight(40)

        self.lbl_file = QLabel("No file uploaded")
        self.lbl_file.setStyleSheet("color: #999999; font-style: italic; font-size: 12px; margin-top: 4px;")
        right_layout.addWidget(self.btn_upload)
        right_layout.addWidget(self.lbl_file)

        right_layout.addStretch()

        # Run button with enhanced styling
        self.btn_run = QPushButton("▶ Run Screening")
        self.btn_run.setGraphicsEffect(self.run_glow)
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setMinimumWidth(180)
        btn_font = QFont()
        btn_font.setPointSize(13)
        btn_font.setWeight(QFont.Weight.Bold)
        self.btn_run.setFont(btn_font)
        right_layout.addWidget(self.btn_run)

        main_layout.addWidget(right_group, stretch=2)

        self.btn_upload.clicked.connect(self.upload_file)
        self.btn_run.clicked.connect(self.run_screening)

    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select SMILES CSV", "", "CSV Files (*.csv)")
        if file_name:
            self.uploaded_csv_path = file_name
            short_name = file_name.split("/")[-1]
            self.lbl_file.setText(f"✓ Selected: {short_name}")
            self.lbl_file.setStyleSheet("color: #10b981; font-style: normal; font-weight: 500; font-size: 12px; margin-top: 4px;")

    def _emit_run(self, payload):
        self.run_screening_signal.emit(payload)

    def _parse_manual_smiles(self):
        raw_text = self.text_input.toPlainText().strip()
        if not raw_text:
            return []
        normalized = raw_text.replace("\n", ",")
        return [token.strip() for token in normalized.split(",") if token.strip()]

    def _collect_property_ranges(self):
        ranges = {}
        for prop in self.properties:
            min_val = self.spinboxes[prop]['min'].value()
            max_val = self.spinboxes[prop]['max'].value()
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            ranges[prop] = (low, high)
        return ranges

    def run_screening(self):
        payload = {
            "csv_path": self.uploaded_csv_path,
            "manual_smiles": self._parse_manual_smiles(),
            "property_ranges": self._collect_property_ranges(),
        }
        self._emit_run(payload)