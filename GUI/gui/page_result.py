import csv
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, 
    QTableWidgetItem, QPushButton, QHeaderView, QFrame, QMessageBox, QFileDialog, QProgressBar
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QFont
from gui.page_visualisation import MoleculeInspectorDialog
from core.inference import compute_transformer_explainability

class ResultsPage(QWidget):
    # Signal to tell the main window to switch back to the input page
    go_back_signal = Signal()

    def __init__(self):
        super().__init__()
        
        self.properties = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 
                           'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
        self.confidence_column_name = "CONF"
        self.confidence_band_column_name = "CONF_BAND"
        self._export_rows = []
        
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
        self.table = QTableWidget(0, len(self.properties) + 3)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(len(self.properties) + 3)
        self.table.setRowCount(0)

        self.table.doubleClicked.connect(self.on_row_double_clicked)

        headers = [
            "SMILES",
            *[p.upper() for p in self.properties],
            self.confidence_column_name,
            self.confidence_band_column_name,
        ]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Set column widths - make SMILES column wider
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(headers) - 1):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(len(headers) - 1, QHeaderView.ResizeToContents)
        
        # Improve row height
        self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.setShowGrid(False)
        
        main_layout.addWidget(self.table)
        
        # Summary Label
        self.summary_label = QLabel("No results yet")
        self.summary_label.setStyleSheet("color: #9ca3af; font-size: 12px; font-style: italic;")
        main_layout.addWidget(self.summary_label)
        
        # Loading Indicator
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate mode
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #1f2329;
            }
            QProgressBar::chunk {
                background-color: #5B6EFF;
                border-radius: 3px;
            }
        """)
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)
        
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
        self.btn_export.clicked.connect(self.export_to_csv)

    def populate_table(self):
        """A temp function to populate the table with dummy data for testing."""
        self.table.setRowCount(0)
        self.table.setRowCount(2)
        self._export_rows = []
        
        # Define proper data
        data = [
            ("CCO", [1.234, 2.567, 3.890, 4.123, 5.456, 6.789, 7.012, 8.345, 9.678, 10.901, 11.234, 12.567]),
            ("c1ccccc1", [5.678, 6.789, 7.890, 8.901, 9.012, 10.123, 11.234, 12.345, 13.456, 14.567, 15.678, 16.789])
        ]
        
        # Add rows to table with proper formatting
        for row_idx, (smiles, values) in enumerate(data):
            self._set_row(row_idx, smiles, [f"{value:.3f}" for value in values], None)
        
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

            explainability = {}
            try:
                explainability = compute_transformer_explainability(smiles)
            except Exception as exc:
                QMessageBox.information(
                    self,
                    "Attention Map Unavailable",
                    f"Could not compute transformer attention map for this molecule.\n\n{exc}",
                )

            self.inspector = MoleculeInspectorDialog(smiles, explainability=explainability)
            self.inspector.exec_()

    def import_from_csv(self, file_path):
        """
        This function will read a CSV file and populate the table with its contents.
        """
        self.table.setRowCount(0)
        self._export_rows = []

        try:
            with open(file_path, "r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                if not reader.fieldnames:
                    raise ValueError("CSV file is missing a header row.")

                smiles_column = self._find_smiles_column(reader.fieldnames)
                if smiles_column is None:
                    raise ValueError("Could not find a SMILES column. Expected one named 'smiles' or 'smile'.")

                row_count = 0
                for row in reader:
                    smiles = (row.get(smiles_column) or "").strip()
                    if not smiles:
                        continue

                    values = [self._extract_property_value(row, prop) for prop in self.properties]
                    self.table.insertRow(row_count)
                    self._set_row(row_count, smiles, values, None)
                    numeric_values = []
                    for value in values:
                        try:
                            numeric_values.append(float(value))
                        except (ValueError, TypeError):
                            numeric_values.append(None)
                    self._export_rows.append(self._build_export_row(smiles, numeric_values, None))
                    row_count += 1

                if row_count == 0:
                    raise ValueError("No valid SMILES rows were found in the CSV.")

                csv_name = Path(file_path).name
                self.summary_label.setText(f"Loaded {row_count} molecule(s) from {csv_name}")
                return True

        except Exception as exc:
            self.summary_label.setText("No results yet")
            QMessageBox.warning(self, "CSV Import Failed", str(exc))
            return False

    def populate_from_predictions(self, predictions, filtered_out=0, failed=0):
        """
        Populate table using [(smiles, [12 regression outputs]), ...].
        """
        self.table.setRowCount(0)
        self._export_rows = []

        for row_idx, row in enumerate(predictions):
            if len(row) == 3:
                smiles, values, confidence = row
            else:
                smiles, values = row
                confidence = None

            formatted = []
            numeric_values = []
            for value in values:
                try:
                    float_value = float(value)
                    formatted.append(f"{float_value:.3f}")
                    numeric_values.append(float_value)
                except (ValueError, TypeError):
                    formatted.append(str(value))
                    numeric_values.append(None)

            self.table.insertRow(row_idx)
            self._set_row(row_idx, smiles, formatted, confidence)
            self._export_rows.append(self._build_export_row(smiles, numeric_values, confidence))

        self.summary_label.setText(
            f"Showing {len(predictions)} molecule(s)"
            f" | filtered out: {filtered_out}"
            f" | failed: {failed}"
        )

    def _find_smiles_column(self, fieldnames):
        normalized = {name.strip().lower(): name for name in fieldnames if name}
        for key in ("smiles", "smile"):
            if key in normalized:
                return normalized[key]
        return fieldnames[0] if fieldnames else None

    def _extract_property_value(self, row, prop):
        candidates = [prop, prop.upper(), prop.capitalize()]
        for candidate in candidates:
            if candidate in row and row[candidate] is not None:
                value = str(row[candidate]).strip()
                if value:
                    try:
                        return f"{float(value):.3f}"
                    except ValueError:
                        return value
        return "-"

    def _set_row(self, row_idx, smiles, values, confidence=None):
        smiles_item = QTableWidgetItem(smiles)
        smiles_item.setFont(QFont('Monaco', 11))
        smiles_item.setForeground(QColor("#e0e0e0"))
        self.table.setItem(row_idx, 0, smiles_item)

        for col_idx, value in enumerate(values, start=1):
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)
            item.setFont(QFont('Montserrat', 11))
            item.setForeground(QColor("#b0b0b0"))
            self.table.setItem(row_idx, col_idx, item)

        conf_col = len(self.properties) + 1
        band_col = len(self.properties) + 2
        conf_text = "-"
        band_text = "UNKNOWN"
        conf_tooltip = "Confidence unavailable"
        band_color = QColor("#9ca3af")
        if isinstance(confidence, dict):
            score = confidence.get("confidence_score")
            used = confidence.get("n_conformers_used")
            requested = confidence.get("n_conformers_requested")
            interval_method = confidence.get("interval_method", "unknown")
            if isinstance(score, (float, int)):
                score_value = float(score)
                conf_text = f"{score_value:.1f}%"
                band_text, band_color = self._classify_confidence_band(score_value)
            conf_tooltip = (
                f"mode: {confidence.get('mode', 'unknown')}\n"
                f"conformers: {used}/{requested}\n"
                f"intervals: {interval_method}\n"
                f"warnings: {len(confidence.get('warnings') or [])}"
            )

        conf_item = QTableWidgetItem(conf_text)
        conf_item.setTextAlignment(Qt.AlignCenter)
        conf_item.setFont(QFont('Montserrat', 11))
        conf_item.setForeground(QColor("#9ecbff"))
        conf_item.setToolTip(conf_tooltip)
        self.table.setItem(row_idx, conf_col, conf_item)

        band_item = QTableWidgetItem(band_text)
        band_item.setTextAlignment(Qt.AlignCenter)
        band_item.setFont(QFont('Montserrat', 11))
        band_item.setForeground(band_color)
        band_item.setToolTip("Confidence band derived from confidence score")
        self.table.setItem(row_idx, band_col, band_item)

    def _classify_confidence_band(self, score):
        if score >= 85.0:
            return "HIGH", QColor("#22c55e")
        if score >= 60.0:
            return "MEDIUM", QColor("#f59e0b")
        return "LOW", QColor("#ef4444")

    def _build_export_row(self, smiles, values, confidence):
        row = {
            "SMILES": smiles,
        }
        for prop, value in zip(self.properties, values):
            row[prop.upper()] = "" if value is None else float(value)

        if not isinstance(confidence, dict):
            row["CONF_SCORE"] = ""
            row["CONF_BAND"] = ""
            row["CONF_MODE"] = ""
            row["CONF_CONFORMERS_USED"] = ""
            row["CONF_CONFORMERS_REQUESTED"] = ""
            row["CONF_WARNINGS"] = ""
            for prop in self.properties:
                base = prop.upper()
                row[f"{base}_STD"] = ""
                row[f"{base}_CV_PCT"] = ""
                row[f"{base}_LOWER"] = ""
                row[f"{base}_UPPER"] = ""
            return row

        score = confidence.get("confidence_score")
        band_text, _ = self._classify_confidence_band(float(score)) if isinstance(score, (float, int)) else ("UNKNOWN", QColor("#9ca3af"))
        row["CONF_SCORE"] = "" if score is None else float(score)
        row["CONF_BAND"] = band_text
        row["CONF_MODE"] = confidence.get("mode", "")
        row["CONF_CONFORMERS_USED"] = confidence.get("n_conformers_used", "")
        row["CONF_CONFORMERS_REQUESTED"] = confidence.get("n_conformers_requested", "")
        warnings = confidence.get("warnings") or []
        row["CONF_WARNINGS"] = " | ".join(str(w) for w in warnings)

        std_values = confidence.get("std") or []
        cv_values = confidence.get("cv_percent") or []
        lower_values = confidence.get("interval_lower") or []
        upper_values = confidence.get("interval_upper") or []
        for idx, prop in enumerate(self.properties):
            base = prop.upper()
            row[f"{base}_STD"] = std_values[idx] if idx < len(std_values) else ""
            row[f"{base}_CV_PCT"] = cv_values[idx] if idx < len(cv_values) else ""
            row[f"{base}_LOWER"] = lower_values[idx] if idx < len(lower_values) else ""
            row[f"{base}_UPPER"] = upper_values[idx] if idx < len(upper_values) else ""
        return row

    def export_to_csv(self):
        if self.table.rowCount() == 0:
            QMessageBox.information(self, "Export", "There are no rows to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results CSV",
            "screening_results.csv",
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        if not file_path.lower().endswith(".csv"):
            file_path = f"{file_path}.csv"

        try:
            with open(file_path, "w", encoding="utf-8", newline="") as csv_file:
                if self._export_rows:
                    confidence_headers = [
                        "SMILES",
                        *[p.upper() for p in self.properties],
                        "CONF_SCORE",
                        "CONF_BAND",
                        "CONF_MODE",
                        "CONF_CONFORMERS_USED",
                        "CONF_CONFORMERS_REQUESTED",
                        "CONF_WARNINGS",
                    ]
                    uncertainty_headers = []
                    for prop in self.properties:
                        base = prop.upper()
                        uncertainty_headers.extend([
                            f"{base}_STD",
                            f"{base}_CV_PCT",
                            f"{base}_LOWER",
                            f"{base}_UPPER",
                        ])
                    headers = confidence_headers + uncertainty_headers

                    writer = csv.DictWriter(csv_file, fieldnames=headers)
                    writer.writeheader()
                    for row_data in self._export_rows:
                        writer.writerow(row_data)
                else:
                    writer = csv.writer(csv_file)
                    headers = []
                    for col in range(self.table.columnCount()):
                        header_item = self.table.horizontalHeaderItem(col)
                        headers.append(header_item.text() if header_item else f"column_{col}")
                    writer.writerow(headers)
                    for row in range(self.table.rowCount()):
                        row_data = []
                        for col in range(self.table.columnCount()):
                            item = self.table.item(row, col)
                            row_data.append(item.text() if item else "")
                        writer.writerow(row_data)

            self.summary_label.setText(
                f"Exported {self.table.rowCount()} molecule(s) to {Path(file_path).name}"
            )
            QMessageBox.information(self, "Export Complete", f"Saved results to:\n{file_path}")
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))