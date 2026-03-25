from PySide6.QtWidgets import QMainWindow, QStackedWidget
from PySide6.QtCore import QEasingCurve
from PySide6.QtGui import QIcon

from gui.page_input import InputPage
from gui.page_result import ResultsPage
from gui.effects import Animation

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
        self.page_input = InputPage()
        self.page_results = ResultsPage()

        # Add pages to the stack
        self.stack.addWidget(self.page_input)    # Index 0
        self.stack.addWidget(self.page_results)  # Index 1

        # Wire up the navigation logic
        self.page_input.run_screening_signal.connect(self.show_results)
        self.page_results.go_back_signal.connect(self.show_input)

        # Keep animation references alive during playback.
        self.anim_input = None
        self.anim_result = None
        self.setWindowOpacity(1.0)

    def show_results(self):
        # Later, you will launch your ONNX thread here. 
        # For now, we just mock the data and switch the page.
        self.page_results.populate_table()
        self.stack.setCurrentIndex(1)

        self.anim_result = Animation(target=self,
                              property_name=b"windowOpacity",
                              duration=300, start_value=0.85, end_value=1.0,
                              easing_curve=QEasingCurve.OutQuad)
        self.anim_result.animation.start()

    def show_input(self):
        self.stack.setCurrentIndex(0)

        self.anim_input = Animation(target=self,
                              property_name=b"windowOpacity",
                              duration=300, start_value=0.85, end_value=1.0,
                              easing_curve=QEasingCurve.OutQuad)
        self.anim_input.animation.start()