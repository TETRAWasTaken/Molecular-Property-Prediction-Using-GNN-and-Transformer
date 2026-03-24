from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QScrollArea, QLabel, QTextEdit, QPushButton, QFileDialog,
    QGraphicsDropShadowEffect
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Signal
from gui.effects import Shadow, Glow

class InputPage(QWidget):
    """
    This class represents the input page of the GUI, where users can input parameters for the simulation.
    """

    run_screening_signal = Signal()


    def __init__(self):
        super().__init__()

        self.properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                           'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
    
        main_layout = QHBoxLayout()

        # Keep distinct effect instances alive per widget.
        self.left_shadow = Shadow(color=QColor(255, 255, 255, 30),
                                  blur_radius=35,
                                  x_offset=0,
                                  y_offset=0).effect
        self.right_shadow = Shadow(color=QColor(255, 255, 255, 30),
                                   blur_radius=35,
                                   x_offset=0,
                                   y_offset=0).effect
        self.upload_glow = Glow(color=QColor(13, 22, 97, 100),
                                blur_radius=25,
                                x_offset=0,
                                y_offset=0).glow
        self.run_glow = Glow(color=QColor(13, 22, 97, 100),
                             blur_radius=25,
                             x_offset=0,
                             y_offset=0).glow

        ########## Column 1 : Property Ranges ##########
        left_group = QGroupBox("Define Target Property Ranges")
        left_group.setGraphicsEffect(self.left_shadow)

        left_layout = QVBoxLayout(left_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        form_layout = QFormLayout(scroll_content)

        self.spinboxes = {}
        for prop in self.properties:
            min_box = QDoubleSpinBox()
            max_box = QDoubleSpinBox()
            min_box.setRange(-1000, 1000)
            max_box.setRange(-1000, 1000)
            min_box.setValue(100)
            max_box.setValue(100)

            row_layout = QHBoxLayout()
            row_layout.addWidget(min_box)
            row_layout.addWidget(QLabel(" - "))
            row_layout.addWidget(max_box)

            form_layout.addRow(f"{prop.upper()}:", row_layout)
            self.spinboxes[prop] = {'min': min_box, 'max': max_box}

        scroll_area.setWidget(scroll_content)
        left_layout.addWidget(scroll_area)
        main_layout.addWidget(left_group, stretch=1)

        ########## Column 2 : Smiles Inputs ##########
        # Manual input of SMILES strings
        right_group = QGroupBox("Input SMILES Strings")
        right_group.setGraphicsEffect(self.right_shadow)
        right_layout = QVBoxLayout(right_group)

        right_layout.addWidget(QLabel("Enter COMMA-SEPARATED SMILES:"))
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("e.g. CCO, c1ccccc1, CC(=O)O")
        right_layout.addWidget(self.text_input)

        right_layout.addWidget(QLabel("- OR -"))

        # File upload for SMILES strings
        self.btn_upload = QPushButton("Upload CSV File")
        self.btn_upload.setGraphicsEffect(self.upload_glow)


        self.lbl_file = QLabel("No file uploaded")
        self.lbl_file.setStyleSheet("color: gray; font-style: italic;")
        right_layout.addWidget(self.btn_upload)
        right_layout.addWidget(self.lbl_file)


        right_layout.addStretch()

        self.btn_run = QPushButton("Run Screening")
        self.btn_run.setGraphicsEffect(self.run_glow)
        
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setStyleSheet("font-size: bold; font-size: 14px;")
        right_layout.addWidget(self.btn_run)

        main_layout.addWidget(right_group, stretch=2)

        self.btn_upload.clicked.connect(self.upload_file)
        self.btn_run.clicked.connect(self.run_screening)

        # Attach the composed layout to this page so widgets are rendered.
        self.setLayout(main_layout)

    def upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select SMILES CSV", "", "CSV Files (*.csv)")
        if file_name:
            short_name = file_name.split("/")[-1]
            self.lbl_file.setText(f"Selected: {short_name}")
            self.lbl_file.setStyleSheet("color: green;")

    def _emit_run(self):
        self.run_screening_signal.emit()

    def run_screening(self):
        self._emit_run()