#!/usr/bin/python3

# https://realpython.com/python-pyqt-qthread/
# https://codeloop.org/pyqt5-qprogressbar-with-qthread-practical-example/

# from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.Qt import QProgressDialog
import sys
import time


class LongComputationThread(QThread):
    # Create a counter thread, simulates a long computation
    #change_value = pyqtSignal(int)
    finished = pyqtSignal()

    def run(self):
        self.cnt = 0
        while self.cnt < 100:
            self.cnt+=1
            time.sleep(0.1)
            # self.change_value.emit(self.cnt)
        self.finished.emit()


class Window(QDialog):
    def __init__(self):
        super().__init__()

        # Create widgets
        self.setWindowTitle("PyQt5 Thread")
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Mal sehen, was passiert"))
        self.button = QPushButton("Start Thread")
        self.button.clicked.connect(self.startProgressBar)
        vbox.addWidget(self.button)
        self.setLayout(vbox)

        self.show()

    def startProgressBar(self):
        self.progressDialog = QProgressDialog("Gaanz lang ...", "Cancel", 0, 100, self)
        self.progressDialog.setWindowModality(Qt.WindowModal)
        self.progressDialog.canceled.connect(self.onCanceled)

        # Start thread
        self.thread = LongComputationThread()
        #self.thread.change_value.connect(self.setProgressVal)
        self.thread.finished.connect(self.onThreadFinished)
        self.thread.start()

        # Start monitor timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimer)
        self.timer.start(1000)

    def setProgressVal(self, val):
        self.progressDialog.setValue(val)

    def onCanceled(self):
        self.timer.stop()
        self.thread.
        print("Canceled")

    def onTimer(self):
        print(self.thread.cnt)
        self.progressDialog.setValue(self.thread.cnt)

    def onThreadFinished(self):
        self.timer.stop()
        print("Thread finished")


if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = Window()
    sys.exit(App.exec_())
