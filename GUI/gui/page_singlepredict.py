from pathlib import Path

from PySide6.QtCore import Qt, Signal, QThread, QUrl
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
	QAbstractItemView,
	QGroupBox,
	QHBoxLayout,
	QHeaderView,
	QLabel,
	QLineEdit,
	QMessageBox,
	QProgressBar,
	QPushButton,
	QCheckBox,
	QTableWidget,
	QTableWidgetItem,
	QVBoxLayout,
	QWidget,
)
from PySide6.QtWebEngineWidgets import QWebEngineView
from rdkit import Chem

from GUI.core.inference import compute_transformer_explainability, run_hybrid_regression_with_confidence
from GUI.core.visualisation import generate_3d_molecule_html_file
from GUI.gui.effects import Glow, Shadow


SINGLE_BUTTON_STYLE = """
	QPushButton {
		background-color: #CC5500;
		color: #FAF9F6;
		border: none;
		border-radius: 8px;
		padding: 10px 16px;
		font-weight: 700;
		letter-spacing: 0.4px;
	}
	QPushButton:hover {
		background-color: #B34700;
	}
	QPushButton:pressed {
		background-color: #8F3B00;
	}
	QPushButton:disabled {
		background-color: #D9B49A;
		color: #F8F2EC;
	}
"""


class SinglePredictThread(QThread):
	finished = Signal(dict)
	error = Signal(str)

	def __init__(self, smiles, n_conformers=3, model_path=None):
		super().__init__()
		self.smiles = (smiles or "").strip()
		self.n_conformers = int(n_conformers)
		self.model_path = model_path

	def run(self):
		if not self.smiles:
			self.error.emit("SMILES input is required")
			return

		try:
			result = run_hybrid_regression_with_confidence(
				self.smiles,
				model_path=self.model_path,
				n_conformers=self.n_conformers,
			)
			payload = {
				"smiles": self.smiles,
				"prediction": result["prediction"].tolist(),
				"confidence": result.get("confidence") or {},
			}
			try:
				payload["explainability"] = compute_transformer_explainability(self.smiles)
			except Exception:
				payload["explainability"] = {}
			self.finished.emit(payload)
		except Exception as exc:
			self.error.emit(str(exc))


class SinglePredictPage(QWidget):
	go_home_signal = Signal()

	def __init__(self):
		super().__init__()

		self.properties = [
			"mu",
			"alpha",
			"homo",
			"lumo",
			"gap",
			"r2",
			"zpve",
			"u0",
			"u298",
			"h298",
			"g298",
			"cv",
		]
		self.worker = None
		self._current_smiles = ""
		self._latest_explainability = {}
		self._last_visualization_html_path = None

		root_layout = QVBoxLayout(self)
		root_layout.setContentsMargins(16, 16, 16, 16)
		root_layout.setSpacing(14)

		title = QLabel("Single Molecule Prediction")
		title_font = QFont()
		title_font.setPointSize(40)
		title_font.setWeight(QFont.Weight.Bold)
		title.setFont(title_font)
		title.setStyleSheet("color: #6C3B1E; font-size: 40px; font-weight: 800; letter-spacing: 0.5px;")
		root_layout.addWidget(title)

		subtitle = QLabel("Predict a single SMILES string and inspect the molecule in the same view.")
		subtitle.setStyleSheet("color: #7A6657; font-size: 16px;")
		root_layout.addWidget(subtitle)

		content_row = QHBoxLayout()
		content_row.setSpacing(14)
		root_layout.addLayout(content_row)

		controls_box = QGroupBox("Input and Prediction")
		controls_box.setGraphicsEffect(
			Shadow(color=QColor(122, 90, 66, 40), blur_radius=20, x_offset=0, y_offset=0).effect
		)
		controls_layout = QVBoxLayout(controls_box)
		controls_layout.setSpacing(10)

		self.smiles_input = QLineEdit()
		self.smiles_input.setPlaceholderText("Enter one SMILES string, e.g. CCO")
		self.smiles_input.setMinimumHeight(38)

		self.btn_predict = QPushButton("Predict Properties")
		self.btn_predict.setMinimumHeight(44)
		self.btn_predict.setStyleSheet(SINGLE_BUTTON_STYLE)
		self.btn_predict.setGraphicsEffect(
			Glow(color=QColor(204, 85, 0, 140), blur_radius=20, x_offset=0, y_offset=0).glow
		)

		self.btn_home = QPushButton("Home")
		self.btn_home.setMinimumHeight(38)
		self.btn_home.setStyleSheet(SINGLE_BUTTON_STYLE)

		self.status_label = QLabel("Enter a SMILES string to start.")
		self.status_label.setWordWrap(True)
		self.status_label.setStyleSheet("color: #7A6657; font-size: 12px;")

		self.progress_bar = QProgressBar()
		self.progress_bar.setRange(0, 0)
		self.progress_bar.setTextVisible(False)
		self.progress_bar.setFixedHeight(6)
		self.progress_bar.setStyleSheet(
			"""
			QProgressBar {
				border: none;
				border-radius: 3px;
				background-color: #F3E8DE;
			}
			QProgressBar::chunk {
				background-color: #CC5500;
				border-radius: 3px;
			}
			"""
		)
		self.progress_bar.hide()

		self.summary_label = QLabel("No prediction yet")
		self.summary_label.setWordWrap(True)
		self.summary_label.setStyleSheet("color: #7A6657; font-size: 12px;")

		self.result_table = QTableWidget(0, 4)
		self.result_table.setHorizontalHeaderLabels(["PROPERTY", "VALUE", "STD", "INTERVAL"])
		self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
		self.result_table.setSelectionMode(QAbstractItemView.NoSelection)
		self.result_table.setAlternatingRowColors(True)
		self.result_table.verticalHeader().setVisible(False)
		self.result_table.horizontalHeader().setStretchLastSection(True)
		self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
		self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
		self.result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
		self.result_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
		self.result_table.setStyleSheet(
			"""
			QTableWidget {
				background-color: #FFFDF8;
				alternate-background-color: #F6EEE4;
				gridline-color: #E2CDBC;
				border: 1px solid #E0C7B1;
				border-radius: 8px;
			}
			QTableWidget::item {
				padding: 8px;
				border-bottom: 1px solid #E8D8CA;
				color: #4A3A2A;
			}
			QHeaderView::section {
				background-color: #CC5500;
				color: #FAF9F6;
				padding: 10px;
				border: none;
				border-bottom: 2px solid #B34700;
				font-weight: 600;
				font-size: 12px;
			}
			"""
		)

		controls_layout.addWidget(QLabel("SMILES"))
		controls_layout.addWidget(self.smiles_input)
		controls_layout.addWidget(self.btn_predict)
		controls_layout.addWidget(self.btn_home)
		controls_layout.addWidget(self.status_label)
		controls_layout.addWidget(self.progress_bar)
		controls_layout.addWidget(self.summary_label)
		controls_layout.addWidget(self.result_table)

		visual_box = QGroupBox("3D Visualization")
		visual_box.setGraphicsEffect(
			Shadow(color=QColor(255, 255, 255, 18), blur_radius=20, x_offset=0, y_offset=0).effect
		)
		visual_layout = QVBoxLayout(visual_box)
		visual_layout.setSpacing(10)

		self.visual_label = QLabel("The molecular view will appear here after prediction.")
		self.visual_label.setStyleSheet("color: #7A6657; font-size: 12px;")
		self.visual_label.setWordWrap(True)

		self.attention_toggle = QCheckBox("Highlight transformer-focused bonds")
		self.attention_toggle.setChecked(False)
		self.attention_toggle.setStyleSheet("color: #7A6657; font-size: 12px;")
		self.attention_toggle.toggled.connect(self._refresh_visualization)

		self.browser = QWebEngineView()
		self.browser.setMinimumWidth(520)
		self.browser.setHtml(self._placeholder_html())

		visual_layout.addWidget(self.visual_label)
		visual_layout.addWidget(self.attention_toggle)
		visual_layout.addWidget(self.browser, stretch=1)

		content_row.addWidget(controls_box, stretch=1)
		content_row.addWidget(visual_box, stretch=1)

		self.btn_predict.clicked.connect(self.start_prediction)
		self.btn_home.clicked.connect(self.go_home_signal.emit)
		self.smiles_input.returnPressed.connect(self.start_prediction)

	def _placeholder_html(self):
		return """
		<html>
		  <body style="background:#FAF9F6;font-family:Montserrat, sans-serif;color:#7A6657;display:flex;align-items:center;justify-content:center;height:100%;margin:0;">
		    <div style="text-align:center;max-width:420px;padding:24px;">
		      <div style="font-size:22px;font-weight:700;color:#6C3B1E;margin-bottom:10px;">3D view pending</div>
		      <div>Enter a valid SMILES string and run prediction to render the molecule here.</div>
		    </div>
		  </body>
		</html>
		"""

	def _set_running(self, running):
		self.btn_predict.setEnabled(not running)
		self.smiles_input.setEnabled(not running)
		self.btn_home.setEnabled(not running)
		if running:
			self.progress_bar.show()
			self.status_label.setText("Running single-molecule prediction...")
			self.btn_predict.setText("Predicting...")
		else:
			self.progress_bar.hide()
			self.btn_predict.setText("Predict Molecule")

	def start_prediction(self):
		if self.worker is not None and self.worker.isRunning():
			return

		smiles = self.smiles_input.text().strip()
		if not smiles:
			QMessageBox.information(self, "SMILES Required", "Please enter a SMILES string.")
			return

		if Chem.MolFromSmiles(smiles) is None:
			QMessageBox.warning(self, "Invalid SMILES", "The SMILES string could not be parsed.")
			return

		self._current_smiles = smiles
		self._latest_explainability = {}
		self.summary_label.setText("Prediction pending...")
		self.visual_label.setText("Rendering molecular structure...")
		self._load_visualization_file(smiles)
		self._set_running(True)

		self.worker = SinglePredictThread(smiles=smiles, n_conformers=3)
		self.worker.finished.connect(self._on_prediction_finished)
		self.worker.error.connect(self._on_prediction_error)
		self.worker.start()

	def _on_prediction_finished(self, payload):
		self._set_running(False)
		self.worker = None

		smiles = payload.get("smiles", self._current_smiles)
		prediction = payload.get("prediction") or []
		confidence = payload.get("confidence") or {}
		explainability = payload.get("explainability") or {}
		self._latest_explainability = explainability

		self._populate_results(prediction, confidence)
		self.summary_label.setText(
			f"Predicted {len(prediction)} properties for {smiles} | confidence: {self._format_confidence(confidence)}"
		)
		self.visual_label.setText("Interactive 3D structure is shown below.")
		self._refresh_visualization()

	def _on_prediction_error(self, message):
		self._set_running(False)
		self.worker = None
		self.summary_label.setText("No prediction yet")
		self.status_label.setText(f"Prediction failed: {message}")
		QMessageBox.critical(self, "Prediction Error", message)

	def _populate_results(self, prediction, confidence):
		self.result_table.setRowCount(0)
		std_values = confidence.get("std") or []
		lower_values = confidence.get("interval_lower") or []
		upper_values = confidence.get("interval_upper") or []

		for row_idx, prop in enumerate(self.properties):
			value = prediction[row_idx] if row_idx < len(prediction) else None
			self.result_table.insertRow(row_idx)
			self.result_table.setItem(row_idx, 0, QTableWidgetItem(prop.upper()))
			self.result_table.setItem(row_idx, 1, QTableWidgetItem(self._format_number(value)))
			self.result_table.setItem(row_idx, 2, QTableWidgetItem(self._format_number(std_values[row_idx] if row_idx < len(std_values) else None)))
			if row_idx < len(lower_values) and row_idx < len(upper_values):
				interval_text = f"[{self._format_number(lower_values[row_idx])}, {self._format_number(upper_values[row_idx])}]"
			else:
				interval_text = "-"
			self.result_table.setItem(row_idx, 3, QTableWidgetItem(interval_text))

	def _format_number(self, value):
		try:
			if value is None:
				return "-"
			return f"{float(value):.3f}"
		except (TypeError, ValueError):
			return str(value)

	def _format_confidence(self, confidence):
		score = confidence.get("confidence_score")
		if isinstance(score, (float, int)):
			return f"{float(score):.1f}%"
		return "unavailable"

	def _refresh_visualization(self):
		if not self._current_smiles:
			self.browser.setHtml(self._placeholder_html())
			return

		explainability = self._latest_explainability if self.attention_toggle.isChecked() else {}
		self._load_visualization_file(
			self._current_smiles,
			atom_contributions=explainability.get("atom_scores"),
			attention_bonds=explainability.get("bond_scores"),
			attention_mode=bool(explainability) and self.attention_toggle.isChecked(),
		)

	def _load_visualization_file(
		self,
		smiles,
		atom_contributions=None,
		attention_bonds=None,
		attention_mode=False,
	):
		html_path = generate_3d_molecule_html_file(
			smiles,
			atom_contributions=atom_contributions,
			attention_bonds=attention_bonds,
			attention_mode=attention_mode,
		)
		self._cleanup_visualization_html_file()
		self._last_visualization_html_path = html_path
		self.browser.load(QUrl.fromLocalFile(html_path))

	def _cleanup_visualization_html_file(self):
		if not self._last_visualization_html_path:
			return
		try:
			Path(self._last_visualization_html_path).unlink(missing_ok=True)
		except Exception:
			pass

	def closeEvent(self, event):
		self._cleanup_visualization_html_file()
		super().closeEvent(event)
