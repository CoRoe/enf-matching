import sys

import cv2

from PyQt5.QtWidgets import (QWidget, QMainWindow, QApplication,
                             QVBoxLayout, QLineEdit, QFileDialog, QLabel,
                             QPushButton, QGroupBox, QGridLayout, QCheckBox,
                             QComboBox, QSpinBox, QTabWidget, QDoubleSpinBox,
                             QMenuBar, QAction, QDialog, QMessageBox,
                             QDialogButtonBox, QProgressDialog, QDockWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.Qt import Qt
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QSize, QRect, QMetaObject, QCoreApplication


class FlimmerView(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)


if __name__ == '__main__':
    try:
        #app = FlimmerApp([])
        app = QApplication(sys.argv)
        player = FlimmerView()
        #player.resize(1200, 1000)
        player.show()
        #app.exec()
        sys.exit(app.exec_())
    except MemoryError as e:
        print(e)
