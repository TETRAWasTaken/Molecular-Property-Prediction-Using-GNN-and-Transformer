from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
	QFileDialog,
	QGroupBox,
	QHBoxLayout,
	QHeaderView,
	QLabel,
	QLineEdit,
	QMessageBox,
	QProgressBar,
	QPushButton,
	QSpinBox,
	QAbstractItemView,
	QTableWidget,
	QTableWidgetItem,
	QTextEdit,
	QVBoxLayout,
	QWidget,
)

from gui.effects import Glow, Shadow
from gui.page_visualisation import MoleculeInspectorDialog
from core.inference import compute_transformer_explainability


class SortableTableWidgetItem(QTableWidgetItem):
	"""Table item supporting numeric sorting while preserving display text."""

	def __init__(self, text="", sort_value=None):
		super().__init__(text)
		self.sort_value = sort_value

	def __lt__(self, other):
		if isinstance(other, SortableTableWidgetItem):
			left = self.sort_value
			right = other.sort_value
			if left is not None and right is not None:
				return left < right
			if left is not None and right is None:
				return False
			if left is None and right is not None:
				return True
		return super().__lt__(other)


class SimilarityIndexPage(QWidget):
	"""Workflow page for query-vs-dataset molecular similarity ranking."""

	run_similarity_signal = Signal(object)
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
		self.uploaded_csv_path = None

		root_layout = QVBoxLayout(self)
		root_layout.setContentsMargins(16, 16, 16, 16)
		root_layout.setSpacing(14)

		title = QLabel("Similarity Search")
		title_font = QFont()
		title_font.setPointSize(40)
		title_font.setWeight(QFont.Weight.Bold)
		title.setFont(title_font)
		title.setStyleSheet("color: #f2f4f8; font-size: 40px; font-weight: 800; letter-spacing: 0.5px;")
		root_layout.addWidget(title)

		subtitle = QLabel(
			"Predict properties for one query molecule and return the top most similar molecules from your dataset."
		)
		subtitle.setStyleSheet("color: #9ca3af; font-size: 16px;")
		subtitle.setToolTip("Double-click any result row to open 3D visualization.")
		root_layout.addWidget(subtitle)

		row_layout = QHBoxLayout()
		row_layout.setSpacing(14)

		controls_box = QGroupBox("Inputs")
		controls_box.setGraphicsEffect(
			Shadow(color=QColor(255, 255, 255, 18), blur_radius=20, x_offset=0, y_offset=0).effect
		)
		controls_layout = QVBoxLayout(controls_box)
		controls_layout.setSpacing(10)

		query_label = QLabel("Query Molecule SMILES")
		query_label_font = QFont()
		query_label_font.setWeight(QFont.Weight.DemiBold)
		query_label.setFont(query_label_font)
		self.query_input = QLineEdit()
		self.query_input.setPlaceholderText("e.g. CCO")
		self.query_input.setMinimumHeight(36)

		topk_row = QHBoxLayout()
		topk_label = QLabel("Top-K")
		self.top_k_spin = QSpinBox()
		self.top_k_spin.setRange(1, 100)
		self.top_k_spin.setValue(10)
		self.top_k_spin.setMinimumHeight(32)
		topk_row.addWidget(topk_label)
		topk_row.addWidget(self.top_k_spin)
		topk_row.addStretch()

		csv_label = QLabel("Dataset via CSV")
		csv_label.setFont(query_label_font)
		self.btn_upload = QPushButton("Upload Dataset CSV")
		self.btn_upload.setGraphicsEffect(
			Glow(color=QColor(91, 110, 255, 90), blur_radius=15, x_offset=0, y_offset=0).glow
		)
		self.btn_upload.setMinimumHeight(38)
		self.lbl_file = QLabel("No CSV selected")
		self.lbl_file.setStyleSheet("color: #9ca3af; font-size: 12px;")

		paste_label = QLabel("Dataset via pasted SMILES (comma or newline separated)")
		paste_label.setFont(query_label_font)
		self.dataset_text = QTextEdit()
		self.dataset_text.setPlaceholderText("C1=CC=CC=C1\nCCO\nCC(=O)O")
		self.dataset_text.setMinimumHeight(130)

		self.btn_run = QPushButton("Run Similarity Search")
		self.btn_run.setMinimumHeight(46)
		run_font = QFont()
		run_font.setWeight(QFont.Weight.Bold)
		run_font.setPointSize(12)
		self.btn_run.setFont(run_font)
		self.btn_run.setGraphicsEffect(
			Glow(color=QColor(91, 110, 255, 120), blur_radius=20, x_offset=0, y_offset=0).glow
		)

		self.btn_home = QPushButton("Home")
		self.btn_home.setMinimumHeight(38)

		controls_layout.addWidget(query_label)
		controls_layout.addWidget(self.query_input)
		controls_layout.addLayout(topk_row)
		controls_layout.addSpacing(6)
		controls_layout.addWidget(csv_label)
		controls_layout.addWidget(self.btn_upload)
		controls_layout.addWidget(self.lbl_file)
		controls_layout.addSpacing(6)
		controls_layout.addWidget(paste_label)
		controls_layout.addWidget(self.dataset_text)
		controls_layout.addStretch()
		controls_layout.addWidget(self.btn_run)
		controls_layout.addWidget(self.btn_home)

		output_box = QGroupBox("Similarity Results")
		output_box.setGraphicsEffect(
			Shadow(color=QColor(255, 255, 255, 18), blur_radius=20, x_offset=0, y_offset=0).effect
		)
		output_layout = QVBoxLayout(output_box)
		output_layout.setSpacing(10)

		self.query_properties_label = QLabel("Query properties: -")
		self.query_properties_label.setWordWrap(True)
		self.query_properties_label.setStyleSheet("color: #cfd5df;")

		self.results_table = QTableWidget(0, 6 + len(self.properties))
		self.results_table.setSortingEnabled(True)
		self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
		self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
		self.results_table.setSelectionMode(QTableWidget.SingleSelection)
		self.results_table.setAlternatingRowColors(True)
		self.results_table.setToolTip("Double-click a molecule row to open 3D visualization.")
		self.results_table.doubleClicked.connect(self.on_row_double_clicked)
		headers = [
			"RANK",
			"SMILES",
			"HYBRID",
			"PROP_SIM",
			"FP_SIM",
			"CONF",
			*[name.upper() for name in self.properties],
		]
		self.results_table.setHorizontalHeaderLabels(headers)
		self.results_table.horizontalHeader().setStretchLastSection(False)
		self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

		self.status_label = QLabel("No similarity search executed yet")
		self.status_label.setStyleSheet("color: #9ca3af; font-size: 12px; font-style: italic;")

		self.progress_bar = QProgressBar()
		self.progress_bar.setRange(0, 0)
		self.progress_bar.setTextVisible(False)
		self.progress_bar.setFixedHeight(6)
		self.progress_bar.setStyleSheet(
			"""
			QProgressBar {
				border: none;
				border-radius: 3px;
				background-color: #1f2329;
			}
			QProgressBar::chunk {
				background-color: #5B6EFF;
				border-radius: 3px;
			}
			"""
		)
		self.progress_bar.hide()

		output_layout.addWidget(self.query_properties_label)
		output_layout.addWidget(self.results_table)
		output_layout.addWidget(self.status_label)
		output_layout.addWidget(self.progress_bar)

		row_layout.addWidget(controls_box, stretch=1)
		row_layout.addWidget(output_box, stretch=2)
		root_layout.addLayout(row_layout)

		self.btn_upload.clicked.connect(self.upload_file)
		self.btn_run.clicked.connect(self.run_similarity_search)
		self.btn_home.clicked.connect(self.go_home_signal.emit)

	def upload_file(self):
		file_name, _ = QFileDialog.getOpenFileName(self, "Select SMILES CSV", "", "CSV Files (*.csv)")
		if file_name:
			self.uploaded_csv_path = file_name
			short_name = file_name.split("/")[-1]
			self.lbl_file.setText(f"Selected: {short_name}")
			self.lbl_file.setStyleSheet("color: #10b981; font-size: 12px;")

	def _parse_pasted_smiles(self):
		raw_text = self.dataset_text.toPlainText().strip()
		if not raw_text:
			return []
		normalized = raw_text.replace("\n", ",")
		return [token.strip() for token in normalized.split(",") if token.strip()]

	def run_similarity_search(self):
		query_smiles = self.query_input.text().strip()
		if not query_smiles:
			QMessageBox.information(self, "Query Required", "Please enter a query SMILES string.")
			return

		payload = {
			"query_smiles": query_smiles,
			"csv_path": self.uploaded_csv_path,
			"manual_smiles": self._parse_pasted_smiles(),
			"top_k": int(self.top_k_spin.value()),
		}
		self.run_similarity_signal.emit(payload)

	def set_running_state(self, is_running, dataset_size=0):
		self.btn_run.setEnabled(not is_running)
		self.btn_upload.setEnabled(not is_running)
		self.btn_home.setEnabled(not is_running)
		self.query_input.setEnabled(not is_running)
		self.dataset_text.setEnabled(not is_running)
		self.top_k_spin.setEnabled(not is_running)
		if is_running:
			self.progress_bar.show()
			self.status_label.setText(
				f"Running confidence prediction and similarity ranking for {dataset_size} candidate molecule(s)..."
			)
		else:
			self.progress_bar.hide()

	def set_status_error(self, message):
		self.status_label.setText(message)

	def populate_similarity_results(self, result_payload):
		query_smiles = result_payload.get("query_smiles") or ""
		query_prediction = result_payload.get("query_prediction") or []
		ranked = result_payload.get("ranked_results") or []
		failed_count = int(result_payload.get("failed_count", 0))
		skipped_count = int(result_payload.get("skipped_count", 0))
		total_candidates = int(result_payload.get("total_candidates", 0))

		if query_prediction and len(query_prediction) == len(self.properties):
			pairs = []
			for name, value in zip(self.properties, query_prediction):
				pairs.append(f"{name.upper()}: {float(value):.3f}")
			self.query_properties_label.setText(
				f"Query [{query_smiles}] predicted properties\n" + " | ".join(pairs)
			)
		else:
			self.query_properties_label.setText(f"Query [{query_smiles}] predicted properties unavailable")

		self.results_table.setSortingEnabled(False)
		self.results_table.setRowCount(0)

		for row_idx, row in enumerate(ranked):
			self.results_table.insertRow(row_idx)

			rank_item = SortableTableWidgetItem(str(row_idx + 1), row_idx + 1)
			rank_item.setTextAlignment(Qt.AlignCenter)
			self.results_table.setItem(row_idx, 0, rank_item)

			smiles = str(row.get("smiles", ""))
			self.results_table.setItem(row_idx, 1, QTableWidgetItem(smiles))

			hybrid = float(row.get("hybrid_score", 0.0))
			prop_sim = float(row.get("property_similarity", 0.0))
			fp_sim = float(row.get("fingerprint_similarity", 0.0))
			conf_score = row.get("confidence_score")
			conf_text = "-"
			conf_value = None
			if isinstance(conf_score, (int, float)):
				conf_value = float(conf_score)
				conf_text = f"{conf_value:.1f}%"

			hybrid_item = SortableTableWidgetItem(f"{hybrid:.4f}", hybrid)
			hybrid_item.setTextAlignment(Qt.AlignCenter)
			self.results_table.setItem(row_idx, 2, hybrid_item)

			prop_item = SortableTableWidgetItem(f"{prop_sim:.4f}", prop_sim)
			prop_item.setTextAlignment(Qt.AlignCenter)
			self.results_table.setItem(row_idx, 3, prop_item)

			fp_item = SortableTableWidgetItem(f"{fp_sim:.4f}", fp_sim)
			fp_item.setTextAlignment(Qt.AlignCenter)
			self.results_table.setItem(row_idx, 4, fp_item)

			conf_item = SortableTableWidgetItem(conf_text, conf_value)
			conf_item.setTextAlignment(Qt.AlignCenter)
			self.results_table.setItem(row_idx, 5, conf_item)

			prediction = row.get("prediction") or []
			for idx, value in enumerate(prediction):
				col = 6 + idx
				try:
					numeric_value = float(value)
					item = SortableTableWidgetItem(f"{numeric_value:.3f}", numeric_value)
				except (TypeError, ValueError):
					item = SortableTableWidgetItem(str(value), None)
				item.setTextAlignment(Qt.AlignCenter)
				self.results_table.setItem(row_idx, col, item)

		self.results_table.setSortingEnabled(True)
		self.results_table.sortItems(2, Qt.DescendingOrder)
		self.status_label.setText(
			f"Showing {len(ranked)} ranked molecules from {total_candidates} candidates"
			f" | skipped: {skipped_count}"
			f" | failed: {failed_count}"
		)

	def on_row_double_clicked(self, item):
		row = item.row()
		smiles_item = self.results_table.item(row, 1)
		if smiles_item is None:
			return

		smiles = smiles_item.text().strip()
		if not smiles:
			return

		explainability = {}
		try:
			explainability = compute_transformer_explainability(smiles)
		except Exception as exc:
			QMessageBox.information(
				self,
				"Attention Map Unavailable",
				f"Could not compute transformer attention map for this molecule.\n\n{exc}",
			)

		inspector = MoleculeInspectorDialog(smiles, explainability=explainability)
		inspector.exec_()