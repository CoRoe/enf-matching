import sys

import cv2

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 779)
        MainWindow.setMinimumSize(QtCore.QSize(920, 405))
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setMinimumSize(QtCore.QSize(400, 400))
        self.centralWidget.setBaseSize(QtCore.QSize(800, 600))
        self.centralWidget.setObjectName("centralWidget")

        # here is where I want to put the image.
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 881, 671))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralWidget)

        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget.setMinimumSize(QtCore.QSize(200, 0))
        self.dockWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.dockWidget.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable)
        self.dockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockWidget.setObjectName("dockWidget")

        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")

        self.groupBox = QtWidgets.QGroupBox(self.dockWidgetContents)
        self.groupBox.setGeometry(QtCore.QRect(0, 30, 141, 281))
        self.groupBox.setMinimumSize(QtCore.QSize(90, 0))
        self.groupBox.setObjectName("groupBox")

        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 20, 121, 32))
        self.pushButton.setObjectName("pushButton")

        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 50, 121, 32))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 80, 121, 32))
        self.pushButton_3.setObjectName("pushButton_3")

        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(20, 120, 100, 20))
        self.radioButton.setObjectName("radioButton")

        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(20, 150, 100, 20))
        self.radioButton_2.setObjectName("radioButton_2")

        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_3.setGeometry(QtCore.QRect(20, 180, 100, 20))
        self.radioButton_3.setObjectName("radioButton_3")
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ARCV"))
        self.groupBox.setTitle(_translate("MainWindow", "Settings"))
        self.pushButton.setText(_translate("MainWindow", "Start Recording"))
        self.pushButton_2.setText(_translate("MainWindow", "Stop Recording"))
        self.pushButton_3.setText(_translate("MainWindow", "Quit GUI"))
        self.radioButton.setText(_translate("MainWindow", "Camera 1"))
        self.radioButton_2.setText(_translate("MainWindow", "Camera 2"))
        self.radioButton_3.setText(_translate("MainWindow", "Camera 3"))


#getting the live vid
class Thread(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, *args, **kwargs):
        QtCore.QThread.__init__(self, *args, **kwargs)
        self.flag = False

    def run(self):
        cap1 = cv2.VideoCapture('/home/cro/Videos/Ein leichtes Mädchen.mp4')
        self.flag = True
        while self.flag:
            ret, frame = cap1.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cvt2qt = QtGui.QImage(rgb_image.data, rgb_image.shape[1],
                                      rgb_image.shape[0],
                                      QtGui.QImage.Format_RGB888)
                self.changePixmap.emit(cvt2qt)                         # I don't really understand this yet

    def stop(self):
        self.flag = False


class Prog(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.th = Thread(self)
        self.th.changePixmap.connect(self.setImage)
        self.th.start()

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def closeEvent(self, event):
        self.th.stop()
        self.th.wait()
        super().closeEvent(event)

if __name__=='__main__':
    Program =  QtWidgets.QApplication(sys.argv)
    MyProg = Prog()
    MyProg.show()
    sys.exit(Program.exec_())
