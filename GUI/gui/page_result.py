from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, 
    QTableWidgetItem, QPushButton, QHeaderView
)
from PySide6.QtCore import Signal

class ResultsPage(QWidget):
    # Signal to tell the main window to switch back to the input page
    go_back_signal = Signal()

    def __init__(self):
        super().__init__()
        
        self.properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 
                           'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        
        layout = QVBoxLayout(self)
        
        # Header
        title = QLabel("Shortlisted Molecules")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
        # The Interactive Table
        self.table = QTableWidget(0, len(self.properties) + 1)
        headers = ["SMILES"] + [p.upper() for p in self.properties]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Make the SMILES column stretch to fill empty space
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        # Bottom Buttons
        btn_layout = QHBoxLayout()
        self.btn_back = QPushButton("← Back to Search")
        self.btn_export = QPushButton("Export to CSV")
        
        btn_layout.addWidget(self.btn_back)
        btn_layout.addStretch() # Pushes export button to the right
        btn_layout.addWidget(self.btn_export)
        
        layout.addLayout(btn_layout)

        # Connections
        self.btn_back.clicked.connect(self.go_back_signal.emit)

    #TODO: Implement populate functionality correctly
    def populate_table(self):
        """A temp function to populate the table with dummy data for testing."""
        self.table.setRowCount(2)
        
        # Row 1
        self.table.setItem(0, 0, QTableWidgetItem("CCO"))
        for col in range(1, 13):
            self.table.setItem(0, col, QTableWidgetItem("1.234"))
            
        # Row 2
        self.table.setItem(1, 0, QTableWidgetItem("c1ccccc1"))
        for col in range(1, 13):
            self.table.setItem(1, col, QTableWidgetItem("5.678"))

    #TODO: Implement export functionality correctly