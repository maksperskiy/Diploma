# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(340, 400)
        MainWindow.setMinimumSize(QtCore.QSize(340, 400))
        MainWindow.setMaximumSize(QtCore.QSize(340, 400))
        MainWindow.setBaseSize(QtCore.QSize(340, 410))
        MainWindow.setMouseTracking(False)
        MainWindow.setAcceptDrops(False)
        MainWindow.setWindowOpacity(1.0)
        MainWindow.setStyleSheet("QMainWindow{\n"
"    background-color: #E6E6FF;\n"
"}")
        MainWindow.setWindowFilePath("")
        MainWindow.setIconSize(QtCore.QSize(32, 32))
        MainWindow.setAnimated(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 330, 321, 61))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet("QPushButton{\n"
"    height: 40px;\n"
"    padding-bottom: 5px;\n"
"    border: none;\n"
"    font: 63 24pt \"Bahnschrift SemiBold Condensed\";\n"
"    color: #444444;\n"
"    background-color:#FFF5D7;\n"
"    border-radius: 5px;\n"
"}\n"
"QPushButton:hover{\n"
"    color: #222222;\n"
"    background-color: #FFF9E9;\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color:#bbbbff;\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("QPushButton{\n"
"    height: 40px;\n"
"    padding-bottom: 5px;\n"
"    border: none;\n"
"    font: 63 24pt \"Bahnschrift SemiBold Condensed\";\n"
"    color: #444444;\n"
"    background-color:#FFF5D7;\n"
"    border-radius: 5px;\n"
"}\n"
"QPushButton:hover{\n"
"    color: #222222;\n"
"    background-color: #FFF9E9;\n"
"}\n"
"QPushButton:pressed{\n"
"    background-color:#bbbbff;\n"
"}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(10, 10, 321, 315))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_4.setStyleSheet("color: #444;\n"
"font: 63 12pt \"Bahnschrift SemiBold\";")
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.textEdit = QtWidgets.QTextEdit(self.horizontalLayoutWidget_2)
        self.textEdit.setStyleSheet("QTextEdit{\n"
"     font: 63 12pt \"Bahnschrift\";\n"
"    background-color:#ccccff;\n"
"    border-radius: 10px;\n"
"}\n"
"QTextEdit:hover{\n"
"    background-color:#bbbbff;\n"
"}")
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_2.addWidget(self.textEdit)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_5 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_5.setStyleSheet("color: #444;\n"
"font: 63 12pt \"Bahnschrift SemiBold\";")
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_3.addWidget(self.label_5)
        self.spinBox = QtWidgets.QSpinBox(self.horizontalLayoutWidget_2)
        self.spinBox.setStyleSheet("QSpinBox{\n"
"    height: 32px;\n"
"     font: 63 16pt \"Bahnschrift\";\n"
"    background-color:#ccccff;\n"
"    border-radius: 5px;\n"
"}\n"
"QSpinBox:hover{\n"
"    background-color:#bbbbff;\n"
"}")
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_3.addWidget(self.spinBox)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem1)
        self.label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label.setStyleSheet("color: #444;\n"
"font: 63 12pt \"Bahnschrift SemiBold\";")
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit.setStyleSheet("QLineEdit{\n"
"    height: 25px;\n"
"     font: 63 12pt \"Bahnschrift\";\n"
"    background-color:#ccccff;\n"
"    border-radius: 5px;\n"
"}\n"
"QLineEdit:hover{\n"
"    background-color:#bbbbff;\n"
"}")
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_3.addWidget(self.lineEdit)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem2)
        self.label_2 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_2.setStyleSheet("color: #444;\n"
"font: 63 12pt \"Bahnschrift SemiBold\";\n"
"word-wrap: break-word;")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_2.setStyleSheet("QLineEdit{\n"
"    height: 25px;\n"
"     font: 63 12pt \"Bahnschrift\";\n"
"    background-color:#ccccff;\n"
"    border-radius: 5px;\n"
"}\n"
"QLineEdit:hover{\n"
"    background-color:#bbbbff;\n"
"}")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout_3.addWidget(self.lineEdit_2)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem3)
        self.label_3 = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.label_3.setStyleSheet("color: #444;\n"
"font: 63 12pt \"Bahnschrift SemiBold\";")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_3.setStyleSheet("QLineEdit{\n"
"    height: 25px;\n"
"     font: 63 12pt \"Bahnschrift\";\n"
"    background-color:#ccccff;\n"
"    border-radius: 5px;\n"
"}\n"
"QLineEdit:hover{\n"
"    background-color:#bbbbff;\n"
"}")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.verticalLayout_3.addWidget(self.lineEdit_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Hands detector"))
        self.pushButton.setText(_translate("MainWindow", "start"))
        self.pushButton_2.setText(_translate("MainWindow", "demo"))
        self.label_4.setText(_translate("MainWindow", "Actions"))
        self.label_5.setText(_translate("MainWindow", "WebCemera"))
        self.label.setText(_translate("MainWindow", "Folder"))
        self.lineEdit.setText(_translate("MainWindow", "Data"))
        self.label_2.setText(_translate("MainWindow", "Number of sequences"))
        self.lineEdit_2.setText(_translate("MainWindow", "30"))
        self.label_3.setText(_translate("MainWindow", "Sequence length"))
        self.lineEdit_3.setText(_translate("MainWindow", "40"))

