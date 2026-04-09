from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QStyle, QStyleOption, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy

from gui.effects import Shadow, Glow


class HomePage(QWidget):
    """Entry page that routes users to screening or similarity workflows."""

    open_screening_signal = Signal()
    open_similarity_signal = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("HomePage")
        self.setStyleSheet("""
            #HomePage {
                background-color: #1A1C22;
            }
        """)
        self.pixmap = QPixmap("GUI/assets/page_home_bg.jpg")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(18)

        self.panel_shadow = Shadow(
            color=QColor(255, 255, 255, 18),
            blur_radius=24,
            x_offset=0,
            y_offset=0,
        ).effect

        self.screening_glow = Glow(
            color=QColor(91, 110, 255, 90),
            blur_radius=20,
            x_offset=0,
            y_offset=0,
        ).glow
        
        self.similarity_glow = Glow(
            color=QColor(91, 110, 255, 90),
            blur_radius=20,
            x_offset=0,
            y_offset=0,
        ).glow

        title = QLabel("Molecular Property Prediction\nand Recommendation")
        title_font = QFont()
        title_font.setPointSize(60)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #f2f4f8; font-size: 60px; font-weight: 800; letter-spacing: 0.5px;")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)

        panel = QFrame()
        panel.setGraphicsEffect(self.panel_shadow)
        panel.setMaximumWidth(1000)
        panel.setStyleSheet(
            """
            QFrame {
                background-color: #252a32;
                border: 2px solid #3A3F49;
                border-radius: 16px;
                padding: 24px;
            }
            """
        )

        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(14)

        button_style = """
            QPushButton {
                background-color: #0d1661;
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 12px 22px;
                font-weight: 700;
                letter-spacing: 0.6px;
                font-size: 22px;
            }
            QPushButton:hover {
                background-color: #1521a0;
            }
            QPushButton:pressed {
                background-color: #0f1970;
            }
        """

        self.btn_screening = QPushButton("Bulk Screening")
        self.btn_screening.setMinimumHeight(70)
        self.btn_screening.setMinimumWidth(360)
        self.btn_screening.setStyleSheet(button_style)
        self.btn_screening.setGraphicsEffect(self.screening_glow)
        screening_font = QFont()
        screening_font.setPointSize(16)
        screening_font.setWeight(QFont.Weight.Bold)
        self.btn_screening.setFont(screening_font)

        screening_desc = QLabel(
            "Filter by target range"
        )
        screening_desc.setWordWrap(True)
        screening_desc.setAlignment(Qt.AlignCenter)
        screening_desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        screening_desc.setMinimumHeight(36)
        screening_desc.setStyleSheet("color: #c2c8d1; font-size: 18px;")

        self.btn_similarity = QPushButton("Similarity Search")
        self.btn_similarity.setMinimumHeight(70)
        self.btn_similarity.setMinimumWidth(360)
        self.btn_similarity.setStyleSheet(button_style)
        self.btn_similarity.setGraphicsEffect(self.similarity_glow)
        similarity_font = QFont()
        similarity_font.setPointSize(16)
        similarity_font.setWeight(QFont.Weight.Bold)
        self.btn_similarity.setFont(similarity_font)

        similarity_desc = QLabel(
            "Find Similar Molecules"
        )
        similarity_desc.setWordWrap(True)
        similarity_desc.setAlignment(Qt.AlignCenter)
        similarity_desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        similarity_desc.setMinimumHeight(36)
        similarity_desc.setStyleSheet("color: #c2c8d1; font-size: 18px;")

        panel_layout.addWidget(self.btn_screening)
        panel_layout.addWidget(screening_desc)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(self.btn_similarity)
        panel_layout.addWidget(similarity_desc)

        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(12)

        panel_row = QHBoxLayout()
        panel_row.setContentsMargins(0, 0, 0, 0)
        panel_row.addStretch()
        panel_row.addWidget(panel)
        panel_row.addStretch()
        layout.addLayout(panel_row)
        layout.addStretch()

        self.btn_screening.clicked.connect(self.open_screening_signal.emit)
        self.btn_similarity.clicked.connect(self.open_similarity_signal.emit)

    def paintEvent(self, event):
        """
        Override paint event to ensure arbitrary background
        """
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        try:
            self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)
            if not self.pixmap.isNull():
                painter.drawPixmap(self.rect(), self.pixmap)
        finally:
            painter.end()