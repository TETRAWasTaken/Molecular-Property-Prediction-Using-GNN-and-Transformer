from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QStyle, QStyleOption, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy

from GUI.gui.effects import Shadow, Glow


HOME_BUTTON_STYLE = """
    QPushButton {
        background-color: #CC5500;
        color: #FAF9F6;
        border: none;
        border-radius: 10px;
        padding: 12px 22px;
        font-weight: 700;
        letter-spacing: 0.6px;
        font-size: 22px;
    }
    QPushButton:hover {
        background-color: #B34700;
    }
    QPushButton:pressed {
        background-color: #8F3B00;
    }
"""


class HomePage(QWidget):
    """Entry page that routes users to screening or similarity workflows."""

    open_screening_signal = Signal()
    open_similarity_signal = Signal()
    open_single_predict_signal = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("HomePage")
        self.setStyleSheet("""
            #HomePage {
                background-color: #FAF9F6;
            }
        """)
        self.pixmap = QPixmap("GUI/assets/page_home_bg.png")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 36)
        layout.setSpacing(18)

        self.panel_shadow = Shadow(
            color=QColor(122, 90, 66, 40),
            blur_radius=24,
            x_offset=0,
            y_offset=0,
        ).effect

        self.screening_glow = Glow(
            color=QColor(204, 85, 0, 120),
            blur_radius=20,
            x_offset=0,
            y_offset=0,
        ).glow
        
        self.similarity_glow = Glow(
            color=QColor(204, 85, 0, 120),
            blur_radius=20,
            x_offset=0,
            y_offset=0,
        ).glow

        self.single_predict_glow = Glow(
            color=QColor(204, 85, 0, 120),
            blur_radius=20,
            x_offset=0,
            y_offset=0,
        ).glow

        title = QLabel("Multi-Modal Molecular Property Prediction \n and Recommendation System")
        title_font = QFont()
        title_font.setPointSize(60)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        title.setStyleSheet("color: #6C3B1E; font-size: 60px; font-weight: 800; letter-spacing: 0.5px;")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)

        panel = QFrame()
        panel.setGraphicsEffect(self.panel_shadow)
        panel.setMaximumWidth(760)
        panel.setStyleSheet(
            """
            QFrame {
                background-color: #FFFDF8;
                border: 2px solid #E0C7B1;
                border-radius: 16px;
                padding: 16px;
            }
            """
        )

        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(10)

        self.btn_screening = QPushButton("Screen By Property Range")
        self.btn_screening.setMinimumHeight(70)
        self.btn_screening.setMinimumWidth(360)
        self.btn_screening.setStyleSheet(HOME_BUTTON_STYLE)
        self.btn_screening.setGraphicsEffect(self.screening_glow)
        screening_font = QFont()
        screening_font.setPointSize(16)
        screening_font.setWeight(QFont.Weight.Bold)
        self.btn_screening.setFont(screening_font)


        self.btn_similarity = QPushButton("Search Similar Molecules")
        self.btn_similarity.setMinimumHeight(70)
        self.btn_similarity.setMinimumWidth(360)
        self.btn_similarity.setStyleSheet(HOME_BUTTON_STYLE)
        self.btn_similarity.setGraphicsEffect(self.similarity_glow)
        similarity_font = QFont()
        similarity_font.setPointSize(16)
        similarity_font.setWeight(QFont.Weight.Bold)
        self.btn_similarity.setFont(similarity_font)


        self.btn_single_predict = QPushButton("Single Molecule Prediction")
        self.btn_single_predict.setMinimumHeight(70)
        self.btn_single_predict.setMinimumWidth(360)
        self.btn_single_predict.setStyleSheet(HOME_BUTTON_STYLE)
        self.btn_single_predict.setGraphicsEffect(self.single_predict_glow)
        single_predict_font = QFont()
        single_predict_font.setPointSize(16)
        single_predict_font.setWeight(QFont.Weight.Bold)
        self.btn_single_predict.setFont(single_predict_font)


        panel_layout.addWidget(self.btn_screening)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(self.btn_similarity)
        panel_layout.addSpacing(8)
        panel_layout.addWidget(self.btn_single_predict)

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
        self.btn_single_predict.clicked.connect(self.open_single_predict_signal.emit)

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