import csv

from PySide6.QtWidgets import QMainWindow, QStackedWidget
from PySide6.QtCore import QEasingCurve
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMessageBox

from gui.page_home import HomePage
from gui.page_input import InputPage
from gui.page_result import ResultsPage
from gui.page_similarityindex import SimilarityIndexPage
from gui.effects import Animation
from core.inference import BatchInferenceThread, SimilarityInferenceThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Quantum Screener")
        self.resize(1200, 750)
        
        # Center window on screen
        screen = self.screen()
        geometry = self.frameGeometry()
        center = screen.availableGeometry().center()
        geometry.moveCenter(center)
        self.move(geometry.topLeft())

        # The Stacked Widget manages multiple pages
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Initialize the pages
        self.page_home = HomePage()
        self.page_input = InputPage()
        self.page_results = ResultsPage()
        self.page_similarity = SimilarityIndexPage()

        # Add pages to the stack
        self.stack.addWidget(self.page_home)       # Index 0
        self.stack.addWidget(self.page_input)      # Index 1
        self.stack.addWidget(self.page_results)    # Index 2
        self.stack.addWidget(self.page_similarity) # Index 3

        # Wire up the navigation logic
        self.page_home.open_screening_signal.connect(self.show_input)
        self.page_home.open_similarity_signal.connect(self.show_similarity)
        self.page_input.go_home_signal.connect(self.show_home)
        self.page_input.run_screening_signal.connect(self.show_results)
        self.page_results.go_back_signal.connect(self.show_input)
        self.page_similarity.go_home_signal.connect(self.show_home)
        self.page_similarity.run_similarity_signal.connect(self.run_similarity_search)

        # Keep animation references alive during playback.
        self.anim_home = None
        self.anim_input = None
        self.anim_result = None
        self.inference_thread = None
        self.similarity_thread = None
        self.active_property_ranges = {}
        self.setWindowOpacity(1.0)

    def show_results(self, payload=None):
        if self._is_any_job_running():
            return

        payload = payload if isinstance(payload, dict) else {}
        csv_path = payload.get("csv_path")
        manual_smiles = payload.get("manual_smiles") or []
        self.active_property_ranges = payload.get("property_ranges") or {}

        csv_smiles = []
        if csv_path:
            try:
                csv_smiles = self._extract_smiles_from_csv(csv_path)
            except Exception as exc:
                QMessageBox.warning(self, "CSV Error", str(exc))
                return

        all_smiles = self._deduplicate_smiles(list(manual_smiles) + csv_smiles)
        if not all_smiles:
            QMessageBox.information(self, "No Input", "Please provide SMILES input or upload a CSV file.")
            return

        self.page_results.table.setRowCount(0)
        self.page_results.summary_label.setText(f"Running ONNX regression for {len(all_smiles)} molecule(s)...")
        self.stack.setCurrentIndex(2)

        self._set_run_state(is_running=True)
        self.inference_thread = BatchInferenceThread(
            all_smiles,
            enable_confidence=True,
            n_conformers=3,
        )
        self.inference_thread.finished.connect(self._on_inference_finished)
        self.inference_thread.error.connect(self._on_inference_error)
        self.inference_thread.start()

    def _on_inference_finished(self, results, failures):
        failed_count = len(failures)
        filtered_predictions = []
        filtered_out_count = 0

        for row in results:
            if len(row) == 3:
                smiles, values, confidence = row
            else:
                smiles, values = row
                confidence = None

            if self._within_selected_ranges(values):
                filtered_predictions.append((smiles, values, confidence))
            else:
                filtered_out_count += 1

        self.page_results.populate_from_predictions(
            predictions=filtered_predictions,
            filtered_out=filtered_out_count,
            failed=failed_count,
        )

        self.anim_result = Animation(target=self,
                              property_name=b"windowOpacity",
                              duration=300, start_value=0.85, end_value=1.0,
                              easing_curve=QEasingCurve.OutQuad)
        self.anim_result.animation.start()

        self._set_run_state(is_running=False)
        self.inference_thread = None

    def _on_inference_error(self, error_message):
        self._set_run_state(is_running=False)
        self.inference_thread = None
        QMessageBox.critical(self, "Inference Error", error_message)

    def _set_run_state(self, is_running):
        self.page_input.btn_run.setEnabled(not is_running)
        self.page_input.btn_home.setEnabled(not is_running)
        if is_running:
            self.page_input.btn_run.setText("Running...")
            self.page_results.progress_bar.show()
            self.page_results.btn_export.setEnabled(False)
            self.page_results.btn_back.setEnabled(False)
        else:
            self.page_input.btn_run.setText("▶ Run Screening")
            self.page_results.progress_bar.hide()
            self.page_results.btn_export.setEnabled(True)
            self.page_results.btn_back.setEnabled(True)

    def _within_selected_ranges(self, values):
        if not self.active_property_ranges:
            return True

        for idx, prop in enumerate(self.page_results.properties):
            if prop not in self.active_property_ranges:
                continue
            low, high = self.active_property_ranges[prop]
            value = float(values[idx])
            if value < low or value > high:
                return False
        return True

    def _extract_smiles_from_csv(self, csv_path):
        smiles_list = []
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames:
                raise ValueError("CSV file must include a header row.")

            smiles_column = self._find_smiles_column(reader.fieldnames)
            if smiles_column is None:
                raise ValueError("CSV must contain a SMILES column (smiles/smile).")

            for row in reader:
                smiles = (row.get(smiles_column) or "").strip()
                if smiles:
                    smiles_list.append(smiles)

        return smiles_list

    def _find_smiles_column(self, fieldnames):
        normalized = {name.strip().lower(): name for name in fieldnames if name}
        for key in ("smiles", "smile", "canonical_smiles", "isomeric_smiles"):
            if key in normalized:
                return normalized[key]
        return None

    def _deduplicate_smiles(self, smiles_list):
        seen = set()
        output = []
        for smiles in smiles_list:
            key = smiles.strip()
            if key and key not in seen:
                seen.add(key)
                output.append(key)
        return output

    def show_input(self):
        self.stack.setCurrentIndex(1)

        self.anim_input = Animation(target=self,
                              property_name=b"windowOpacity",
                              duration=300, start_value=0.85, end_value=1.0,
                              easing_curve=QEasingCurve.OutQuad)
        self.anim_input.animation.start()

    def show_home(self):
        if self._is_any_job_running():
            return
        self.stack.setCurrentIndex(0)

        self.anim_home = Animation(
            target=self,
            property_name=b"windowOpacity",
            duration=300,
            start_value=0.85,
            end_value=1.0,
            easing_curve=QEasingCurve.OutQuad,
        )
        self.anim_home.animation.start()

    def show_similarity(self):
        if self._is_any_job_running():
            return
        self.stack.setCurrentIndex(3)

    def run_similarity_search(self, payload=None):
        if self._is_any_job_running():
            return

        payload = payload if isinstance(payload, dict) else {}
        query_smiles = (payload.get("query_smiles") or "").strip()
        csv_path = payload.get("csv_path")
        manual_smiles = payload.get("manual_smiles") or []
        top_k = int(payload.get("top_k", 10) or 10)

        if not query_smiles:
            QMessageBox.information(self, "Query Required", "Please provide one query SMILES.")
            return

        csv_smiles = []
        if csv_path:
            try:
                csv_smiles = self._extract_smiles_from_csv(csv_path)
            except Exception as exc:
                QMessageBox.warning(self, "CSV Error", str(exc))
                return

        dataset_smiles = self._deduplicate_smiles(list(manual_smiles) + csv_smiles)
        if not dataset_smiles:
            QMessageBox.information(
                self,
                "Dataset Required",
                "Please upload a dataset CSV and/or paste dataset SMILES.",
            )
            return

        self.page_similarity.set_running_state(True, dataset_size=len(dataset_smiles))
        self.similarity_thread = SimilarityInferenceThread(
            query_smiles=query_smiles,
            dataset_smiles=dataset_smiles,
            top_k=top_k,
            n_conformers=3,
            property_weight=0.7,
            fingerprint_weight=0.3,
        )
        self.similarity_thread.finished.connect(self._on_similarity_finished)
        self.similarity_thread.error.connect(self._on_similarity_error)
        self.similarity_thread.start()

    def _on_similarity_finished(self, payload):
        self.page_similarity.set_running_state(False)
        self.page_similarity.populate_similarity_results(payload)
        self.similarity_thread = None

    def _on_similarity_error(self, error_message):
        self.page_similarity.set_running_state(False)
        self.page_similarity.set_status_error(f"Similarity run failed: {error_message}")
        self.similarity_thread = None
        QMessageBox.critical(self, "Similarity Error", error_message)

    def _is_any_job_running(self):
        screening_running = self.inference_thread is not None and self.inference_thread.isRunning()
        similarity_running = self.similarity_thread is not None and self.similarity_thread.isRunning()
        return screening_running or similarity_running