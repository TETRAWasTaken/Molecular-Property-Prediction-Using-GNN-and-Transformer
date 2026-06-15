import csv

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMainWindow, QStackedWidget, QGraphicsOpacityEffect
from PySide6.QtCore import QEasingCurve, QPoint, QParallelAnimationGroup, QPropertyAnimation
from PySide6.QtWidgets import QMessageBox

from GUI.gui.page_home import HomePage
from GUI.gui.page_input import InputPage
from GUI.gui.page_result import ResultsPage
from GUI.gui.page_similarityindex import SimilarityIndexPage
from GUI.gui.page_singlepredict import SinglePredictPage
from GUI.gui.effects import Animation
from GUI.core.inference import BatchInferenceThread, SimilarityInferenceThread

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
        self.page_singlepredict = SinglePredictPage()

        # Add pages to the stack
        self.stack.addWidget(self.page_home)       # Index 0
        self.stack.addWidget(self.page_input)      # Index 1
        self.stack.addWidget(self.page_results)    # Index 2
        self.stack.addWidget(self.page_similarity) # Index 3
        self.stack.addWidget(self.page_singlepredict) # Index 4

        # Wire up the navigation logic
        self.page_home.open_screening_signal.connect(self.show_input)
        self.page_home.open_similarity_signal.connect(self.show_similarity)
        self.page_home.open_single_predict_signal.connect(self.show_single_predict)
        self.page_input.go_home_signal.connect(self.show_home)
        self.page_input.run_screening_signal.connect(self.show_results)
        self.page_results.go_back_signal.connect(self.show_input)
        self.page_similarity.go_home_signal.connect(self.show_home)
        self.page_similarity.run_similarity_signal.connect(self.run_similarity_search)
        self.page_singlepredict.go_home_signal.connect(self.show_home)

        # Keep animation references alive during playback.
        self.anim_home = None
        self.anim_input = None
        self.anim_result = None
        self.transition_group = None
        self.transition_effect = None
        self.is_transition_running = False
        self.pending_page_index = None
        self.inference_thread = None
        self.similarity_thread = None
        self.active_property_ranges = {}
        self.setWindowOpacity(1.0)

    def _wait_for_thread(self, thread):
        if thread is None:
            return
        if isinstance(thread, QThread) and thread.isRunning():
            thread.wait()

    def cleanup_background_tasks(self):
        self._wait_for_thread(self.inference_thread)
        self._wait_for_thread(self.similarity_thread)
        self._wait_for_thread(getattr(self, "engine_warmup_thread", None))
        self._wait_for_thread(getattr(self.page_singlepredict, "worker", None))

    def closeEvent(self, event):
        self.cleanup_background_tasks()
        super().closeEvent(event)

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
        self._transition_to_page(2, direction=1)

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

    def _transition_to_page(self, target_index, direction):
        if target_index == self.stack.currentIndex():
            return

        if self.is_transition_running:
            self.pending_page_index = target_index
            return

        target_page = self.stack.widget(target_index)
        if target_page is None:
            return

        self.is_transition_running = True
        self.pending_page_index = None
        self.stack.setCurrentIndex(target_index)

        stack_rect = self.stack.rect()
        if stack_rect.isNull():
            self.is_transition_running = False
            return

        offset = max(48, stack_rect.width() // 12)
        start_pos = QPoint(offset if direction > 0 else -offset, 0)
        end_pos = QPoint(0, 0)

        target_page.setGeometry(stack_rect)
        target_page.move(start_pos)
        target_page.show()
        target_page.raise_()

        opacity_effect = QGraphicsOpacityEffect(target_page)
        opacity_effect.setOpacity(0.0)
        target_page.setGraphicsEffect(opacity_effect)

        position_animation = QPropertyAnimation(target_page, b"pos", self)
        position_animation.setDuration(220)
        position_animation.setStartValue(start_pos)
        position_animation.setEndValue(end_pos)
        position_animation.setEasingCurve(QEasingCurve.OutCubic)

        opacity_animation = QPropertyAnimation(opacity_effect, b"opacity", self)
        opacity_animation.setDuration(220)
        opacity_animation.setStartValue(0.0)
        opacity_animation.setEndValue(1.0)
        opacity_animation.setEasingCurve(QEasingCurve.OutQuad)

        transition_group = QParallelAnimationGroup(self)
        transition_group.addAnimation(position_animation)
        transition_group.addAnimation(opacity_animation)

        def _finish_transition():
            target_page.setGraphicsEffect(None)
            self.transition_group = None
            self.transition_effect = None
            self.is_transition_running = False
            if self.pending_page_index is not None:
                next_index = self.pending_page_index
                self.pending_page_index = None
                self._transition_to_page(next_index, direction)

        transition_group.finished.connect(_finish_transition)
        self.transition_group = transition_group
        self.transition_effect = opacity_effect
        transition_group.start()

    def show_input(self):
        self._transition_to_page(1, direction=1)

        self.anim_input = Animation(target=self,
                              property_name=b"windowOpacity",
                              duration=300, start_value=0.85, end_value=1.0,
                              easing_curve=QEasingCurve.OutQuad)
        self.anim_input.animation.start()

    def show_home(self):
        if self._is_any_job_running():
            return
        self._transition_to_page(0, direction=-1)

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
        self._transition_to_page(3, direction=1)

    def show_single_predict(self):
        if self._is_any_job_running():
            return
        self._transition_to_page(4, direction=1)

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