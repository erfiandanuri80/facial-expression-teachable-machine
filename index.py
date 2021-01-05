from PyQt5 import QtWidgets, uic, QtSql, QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import capture, main, project, numDataVisual

#CODE UNTUK HALAMAN UTAMA DEKSTOP


class Index(QtWidgets.QMainWindow):
    def __init__(self, num_class):
        super(Index, self).__init__()
        uic.loadUi('index.ui', self)
        num_class = int(num_class)

        #KONDISI JUMLAH INPUTAN CLASS
        if num_class == 2:
            self.label_7.hide()
            self.pushButton_7.hide()
            self.comboBox_3.hide()

            self.label_8.hide()
            self.pushButton_8.hide()
            self.comboBox_4.hide()

            self.label_9.hide()
            self.pushButton_9.hide()
            self.comboBox_5.hide()
        elif num_class == 3:
            self.label_8.hide()
            self.pushButton_8.hide()
            self.comboBox_4.hide()

            self.label_9.hide()
            self.pushButton_9.hide()
            self.comboBox_5.hide()
        elif num_class == 4:
            self.label_9.hide()
            self.pushButton_9.hide()
            self.comboBox_5.hide()

        self.label_22.setText(str(num_class))
        self.pushButton.clicked.connect(self.createDatasetClass1)
        self.pushButton_3.clicked.connect(self.backtoMenu)
        self.pushButton_4.clicked.connect(self.showDataTraining)
        self.pushButton_5.clicked.connect(self.showDataValidation)
        self.pushButton_6.clicked.connect(self.createDatasetClass2)
        self.pushButton_7.clicked.connect(self.createDatasetClass3)
        self.pushButton_8.clicked.connect(self.createDatasetClass4)
        self.pushButton_9.clicked.connect(self.createDatasetClass5)
        self.pushButton_10.clicked.connect(self.trainModel)
        self.pushButton_11.clicked.connect(self.showAccuracyTraining)
        self.pushButton_11.setEnabled(False)
        self.show()

    #MEMBUAT DATASET IMAGE KE CLASS 1
    def createDatasetClass1(self):
        classes = self.label.text()
        nameClass = self.comboBox.currentText()
        print(nameClass, classes)
        capture.capturingFrame(nameClass)

    #MEMBUAT DATASET IMAGE KE CLASS 2
    def createDatasetClass2(self):
        classes = self.label_6.text()
        nameClass = self.comboBox_2.currentText()
        print(nameClass, classes)
        capture.capturingFrame(nameClass)

    #MEMBUAT DATASET IMAGE KE CLASS 3
    def createDatasetClass3(self):
        classes = self.label_7.text()
        nameClass = self.comboBox_3.currentText()
        print(nameClass, classes)
        capture.capturingFrame(nameClass)

    #MEMBUAT DATASET IMAGE KE CLASS 4
    def createDatasetClass4(self):
        classes = self.label_8.text()
        nameClass = self.comboBox_4.currentText()
        print(nameClass, classes)
        capture.capturingFrame(nameClass)

    #MEMBUAT DATASET IMAGE KE CLASS 5
    def createDatasetClass5(self):
        classes = self.label_9.text()
        nameClass = self.comboBox_5.currentText()
        print(nameClass, classes)
        capture.capturingFrame(nameClass)

    #SHOW DISTRIBUSI DATA TRAINING PER CLASS
    def showDataTraining(self):
        numDataVisual.showDataTraining("dataset/")

    #SHOW DISTRIBUSI DATA VALIDATION PER CLASS
    def showDataValidation(self):
        numDataVisual.showDataValidation("dataset/")

    #FMEMANGGIL FUNGSI TRAINING MODEL
    def trainModel(self):
        epochs = self.lineEdit.text()
        batch_size = self.comboBox_6.currentText()
        num_class = self.label_22.text()
        epochs = int(epochs)
        batch_size = int(batch_size)
        num_class = int(num_class)
        project.trainingModel(epochs, batch_size, num_class)
        if project.acc != "":
            self.pushButton_11.setEnabled(True)
            self.label_24.setText("TRAINING MODEL SUKSES")

    #SHOW ACCURACY TRAINING
    #NOTES: MUNGKIN MASIH ADA ERROR
    def showAccuracyTraining(self):
        acc = project.acc
        val_acc = project.val_acc
        epochs = self.lineEdit.text()
        epochs = int(epochs)
        project.graphAccuracy(acc, val_acc, epochs)

    #KEMBALI KE MENU DIALOG
    def backtoMenu(self):
        self.main = main.Main()
        self.close()
        self.main.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Index()
    app.exec_()
