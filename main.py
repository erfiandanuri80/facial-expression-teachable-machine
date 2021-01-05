from PyQt5 import QtWidgets, uic, QtSql, QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import index
import os
from pathlib import Path

## MAIN WINDOW ##
## MAIN APPLICATION ##


class Main(QtWidgets.QDialog):
    def __init__(self):
        super(Main, self).__init__()
        uic.loadUi('main.ui', self)
        self.pushButton.clicked.connect(self.mainprogram)
        self.show()

    def mainprogram(self):
        num_class = self.comboBox.currentText()
        self.index = index.Index(num_class)
        self.close()
        self.index.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    app.exec_()
