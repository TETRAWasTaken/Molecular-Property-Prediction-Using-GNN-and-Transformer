from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, 
    QTableWidgetItem, QPushButton, QHeaderView, QFrame
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QFont
from gui.page_visualisation import MoleculeInspectorDialog

class ResultsPage(QWidget):
    # Signal to tell the main window to switch back to the input page
    go_back_signal = Signal()

    def __init__(self):
        super().__init__()
        
        self.properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 
                           'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # Header Section
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        
        title = QLabel("Shortlisted Molecules")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #F2F4F8; letter-spacing: 0.5px;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Results from molecular screening. Double-click a row to visualize in 3D.")
        subtitle_font = QFont()
        subtitle_font.setPointSize(11)
        subtitle.setFont(subtitle_font)
        subtitle.setStyleSheet("color: #9ca3af;")
        header_layout.addWidget(subtitle)
        
        main_layout.addWidget(header_frame)
        
        # The Interactive Table with enhanced styling
        self.table = QTableWidget(0, len(self.properties) + 1)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(len(self.properties) + 1)
        self.table.setRowCount(0)

        self.table.doubleClicked.connect(self.on_row_double_clicked)

        headers = ["SMILES"] + [p.upper() for p in self.properties]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Set column widths - make SMILES column wider
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(headers)):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        # Improve row height
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setShowGrid(False)
        
        main_layout.addWidget(self.table)
        
        # Summary Label
        self.summary_label = QLabel("No results yet")
        self.summary_label.setStyleSheet("color: #9ca3af; font-size: 12px; font-style: italic;")
        main_layout.addWidget(self.summary_label)
        
        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        self.btn_back = QPushButton("← Back to Search")
        self.btn_back.setMinimumHeight(40)
        self.btn_back.setMinimumWidth(140)
        
        self.btn_export = QPushButton("Export to CSV")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.setMinimumWidth(140)
        
        btn_layout.addWidget(self.btn_back)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_export)
        
        main_layout.addLayout(btn_layout)

        # Connections
        self.btn_back.clicked.connect(self.go_back_signal.emit)

    def populate_table(self):
        """A temp function to populate the table with dummy data for testing."""
        self.table.setRowCount(2)
        
        # Define proper data
        data = [
            ("CCO", [1.234, 2.567, 3.890, 4.123, 5.456, 6.789, 7.012, 8.345, 9.678, 10.901, 11.234, 12.567]),
            ("c1ccccc1", [5.678, 6.789, 7.890, 8.901, 9.012, 10.123, 11.234, 12.345, 13.456, 14.567, 15.678, 16.789])
        ]
        
        # Add rows to table with proper formatting
        for row_idx, (smiles, values) in enumerate(data):
            # Add SMILES column
            smiles_item = QTableWidgetItem(smiles)
            smiles_item.setFont(QFont('Monaco', 11))
            smiles_item.setForeground(QColor("#e0e0e0"))
            self.table.setItem(row_idx, 0, smiles_item)
            
            # Add numerical properties with formatting
            for col_idx, value in enumerate(values, start=1):
                formatted_value = f"{value:.3f}"
                item = QTableWidgetItem(formatted_value)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont('Montserrat', 11))
                item.setForeground(QColor("#b0b0b0"))
                self.table.setItem(row_idx, col_idx, item)
        
        # Update summary
        self.summary_label.setText(f"Showing {self.table.rowCount()} molecule(s)")

    def on_row_double_clicked(self, item):
        """
        This function will be called when a row is double-clicked. 
        You will implement the logic to open the 3D inspector here.
        """
        row = item.row()
        smiles_item = self.table.item(row, 0)

        if smiles_item:
            smiles = smiles_item.text()
            print(f"Double-clicked on SMILES: {smiles}")

            self.inspector = MoleculeInspectorDialog(smiles)
            self.inspector.exec_()



    #TODO: Implement export functionality correctly