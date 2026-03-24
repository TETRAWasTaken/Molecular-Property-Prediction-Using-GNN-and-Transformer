from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QScrollArea, QLabel, QTextEdit, QPushButton, QFileDialog,
    QGraphicsDropShadowEffect
)
from PySide6.QtGui import QColor
from PySide6.QtCore import QPropertyAnimation, Signal

class Shadow:
    """
    This class creates a shadow effect that can be applied to any widget.
    """
    def __init__(self, color=QColor(187, 134, 252, 100),
                blur_radius=35,
                x_offset=0,
                y_offset=0
            ):
        self.effect = QGraphicsDropShadowEffect()
        self.effect.setBlurRadius(blur_radius)
        self.effect.setXOffset(x_offset)
        self.effect.setYOffset(y_offset)
        self.effect.setColor(color)

class Animation:
    def __init__(self,
                 target,
                 property_name=b"opacity",
                 duration=1000,
                 start_value=0, 
                 end_value=100,
                 easing_curve=None):
        self.animation = QPropertyAnimation(target, property_name)
        self.animation.setDuration(duration)
        self.animation.setStartValue(start_value)
        self.animation.setEndValue(end_value)
        if easing_curve:
            self.animation.setEasingCurve(easing_curve)


class Glow:
    """
    This class creates a glow effect that can be applied to any widget.
    """
    def __init__(self, color=QColor(187, 134, 252, 100), 
                 blur_radius=25, 
                 x_offset=0, 
                 y_offset=0
                 ):
        self.glow = QGraphicsDropShadowEffect()
        self.glow.setBlurRadius(blur_radius)
        self.glow.setXOffset(x_offset)
        self.glow.setYOffset(y_offset)
        self.glow.setColor(color)