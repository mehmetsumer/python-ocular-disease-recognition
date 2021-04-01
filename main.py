# -*- coding: utf-8 -*-
import os
import pathlib
import random
import sys
from datetime import datetime

import cv2
import skimage
import seaborn as sns
from joblib import dump, load
from skimage import io
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.uic import *
from skimage.transform import AffineTransform
from sklearn import preprocessing
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.optimizer_v1 import SGD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Flatten, BatchNormalization
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.constraints import maxnorm
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier


class window(QMainWindow):

    def __init__(self):
        super(window, self).__init__()
        loadUi("main.ui", self)
        self.show()
        self.bt_startOver.clicked.connect(self.initFunc)
        self.bt_imgpath.clicked.connect(self.GetImagesPath)
        self.bt_transfer.clicked.connect(self.TransferLearningClick)
        self.bt_split.clicked.connect(self.Split)
        self.bt_train.clicked.connect(self.Train)
        self.bt_csv.clicked.connect(self.GetCSVFile)
        self.cb_kfoldk.currentTextChanged.connect(self.ChangeKfoldK)
        self.bt_chooseImage.clicked.connect(self.TestModel)
        self.cb_kfoldk_2.currentTextChanged.connect(self.ChangeKfoldK_2)
        self.cb_splitalgo.currentTextChanged.connect(self.SetKFoldVisible)
        self.bt_alertCancel.clicked.connect(self.HideAlert)
        self.bt_showOldGraphs.clicked.connect(self.ShowOldGraphs)
        self.bt_saveModel.clicked.connect(self.SaveCurrentModel)
        self.initFunc()

    def initFunc(self):
        self.successPath = "images/success.png"
        self.imPath = "images/"
        self.imagesPath = "images/all/"
        self.classCodes = {"N": "Normal", "D": "Diabetes ", "G": "Glaucoma", "C": "Cataract",
                           "A": "Age related Macular Degeneration", "H": "Hypertension", "M": "Pathological Myopia",
                           "O": "Other diseases/abnormalities"}

        self.classNames = ["Diabetes ", "Glaucoma", "Cataract",
                           "Age related Macular Degeneration", "Hypertension", "Pathological Myopia", "Normal",
                            "Other diseases/abnormalities"]
        self.tabWidget_2.setCurrentIndex(0)
        self.groupBox_2.setEnabled(False)
        self.groupBox_3.setEnabled(False)
        self.groupBox_4.setEnabled(False)
        self.groupBox_5.setEnabled(False)
        self.lbCheck1.setEnabled(False)
        self.lbCheck2.setEnabled(False)
        self.lbCheck3.setEnabled(False)
        self.lbCheck4.setEnabled(False)
        self.bt_saveModel.setEnabled(False)
        self.lbCheck5.setEnabled(False)
        self.label_9.setVisible(False)
        self.cb_k.setVisible(False)
        self.gb_alert.setVisible(False)
        self.holdOutBool = False
        self.TransferLearningDone = False
        self.groupBox.setStyleSheet("QGroupBox { border: 1px solid blue; }")
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid gray; }")
        self.groupBox_3.setStyleSheet("QGroupBox { border: 1px solid gray; }")
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid gray; }")
        self.groupBox_5.setStyleSheet("QGroupBox { border: 1px solid gray; }")
        self.SetSuccessImages()
        self.bt_csv.setEnabled(True)
        self.lb_imgpath.setText("")
        self.bt_imgpath.setEnabled(True)
        self.bt_transfer.setEnabled(True)
        self.cb_transferalgo.setEnabled(True)
        self.maxData.setEnabled(True)
        self.cb_augmentation.setEnabled(True)
        self.bt_split.setEnabled(True)
        self.cb_splitalgo.setEnabled(True)
        self.cb_trainperc.setEnabled(True)
        self.cb_kfoldk.setVisible(False)
        self.label_14.setVisible(False)
        self.cb_kfoldk_2.setVisible(False)
        self.label_13.setVisible(False)
        self.tv_Xtest.clear()
        self.tb_Xtrain.clear()
        self.tv_yTest.clear()
        self.tb_yTrain.clear()
        self.HideAlert()
        self.lb_roc.setText("ROC Eğrisi Grafiği Henüz Getirilmedi")
        self.lb_overlapped.setText("Overlapped Matrix Henüz Getirilmedi")
        self.lb_confusion.setText("Karışıklık Matrisi Henüz Getirilmedi")

    def ShowOldGraphs(self):
        pixmap = QPixmap(self.imPath + 'graphs/roc.png')
        pixmap = pixmap.scaled(self.lb_roc.width(), self.lb_roc.height())
        self.lb_roc.setPixmap(pixmap)
        pixmap = QPixmap(self.imPath + 'graphs/overlapped.png')
        pixmap = pixmap.scaled(self.lb_overlapped.width(), self.lb_overlapped.height())
        self.lb_overlapped.setPixmap(pixmap)
        pixmap = QPixmap(self.imPath + 'graphs/confusion_hold-out.png')
        pixmap = pixmap.scaled(self.lb_confusion.width(), self.lb_confusion.height())
        self.lb_confusion.setPixmap(pixmap)
        images = os.listdir(self.imPath + "graphs")
        for i, path in enumerate(images):
            if "fold" in path:
                fold = path.split("_")[2][0]
                print(fold)
                im = cv2.imread(self.imPath + "graphs/" + path)
                if im is not None:
                    cv2.imshow("Fold-" + str(fold), im)

        self.ShowAlert("K-Fold Confusion Matrixler ayrı ayrı pencerede gösterildi.")

    def HideAlert(self):
        self.gb_alert.setVisible(False)

    def ShowAlert(self, text, type="success"):
        self.lb_alert.setText(text)
        if type == "success":
            self.gb_alert.setStyleSheet("color: #02c20f;background-color: #e7f9f1;border: 1px solid #02c20f;")
            self.lb_alert.setStyleSheet("color: #02c20f;background-color: #e7f9f1;border:none;")
        elif type == "error":
            self.gb_alert.setStyleSheet("color: #ff000d;background-color: #faeded;border: 1px solid #ff000d;")
            self.lb_alert.setStyleSheet("color: #ff000d;background-color: #faeded;border:none;")

        lines = len(text.splitlines())
        if lines == 1:
            lines = 50
        else:
            lines = 50 + lines * 14
        self.gb_alert.setGeometry((self.width() - self.gb_alert.width() - 8), 10, self.gb_alert.width(), lines)
        self.lb_alert.setGeometry(self.lb_alert.x(), self.lb_alert.y(), self.lb_alert.width(),
                                  (self.gb_alert.height() - 31))
        self.gb_alert.setVisible(True)

    def SetKFoldVisible(self):
        if self.cb_splitalgo.currentText() == "Hold Out":
            self.label_9.setVisible(False)
            self.cb_k.setVisible(False)
            self.cb_kfoldk.setVisible(False)
            self.label_14.setVisible(False)
            self.cb_kfoldk_2.setVisible(False)
            self.label_13.setVisible(False)
            self.label_8.setVisible(True)
            self.cb_trainperc.setVisible(True)
        else:
            self.label_9.setVisible(True)
            self.cb_k.setVisible(True)
            self.cb_kfoldk.setVisible(True)
            self.label_14.setVisible(True)
            self.cb_kfoldk_2.setVisible(True)
            self.label_13.setVisible(True)
            self.label_8.setVisible(False)
            self.cb_trainperc.setVisible(False)

    def SetSuccessImages(self):
        pixmap = QPixmap(self.successPath)
        pixmap = pixmap.scaled(61, 61, Qt.KeepAspectRatio)
        self.lbCheck1.setPixmap(pixmap)
        pixmap = QPixmap(self.successPath)
        pixmap = pixmap.scaled(61, 61, Qt.KeepAspectRatio)
        self.lbCheck2.setPixmap(pixmap)
        pixmap = QPixmap(self.successPath)
        pixmap = pixmap.scaled(61, 61, Qt.KeepAspectRatio)
        self.lbCheck3.setPixmap(pixmap)
        pixmap = QPixmap(self.successPath)
        pixmap = pixmap.scaled(61, 61, Qt.KeepAspectRatio)
        self.lbCheck4.setPixmap(pixmap)
        pixmap = QPixmap(self.successPath)
        pixmap = pixmap.scaled(61, 61, Qt.KeepAspectRatio)
        self.lbCheck5.setPixmap(pixmap)

    def GetImagesPath(self):
        file = QFileDialog.getExistingDirectory(self, 'Choose a Directory', str(pathlib.Path().absolute()) + "\images")
        if (len(file) == 0 or file == "") and self.lbCheck1.isEnabled():
            return
        elif len(file) == 0 or file == "":
            self.ShowAlert("Lütfen dosya seçiniz", "error")
            self.groupBox.setStyleSheet("QGroupBox { border: 1px solid red;}")
            return
        self.HideAlert()
        self.imagesPath = str(file) + "/"
        self.lb_imgpath.setText("Yol: " + self.imagesPath)
        self.lbCheck1.setEnabled(True)
        self.groupBox.setStyleSheet("QGroupBox { border: 1px solid green; }")
        self.groupBox_2.setEnabled(True)
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid blue; }")

    def GetCSVFile(self):
        file = QFileDialog.getOpenFileName(self, 'Choose a CSV File',
                                           str(pathlib.Path().absolute()),
                                           "Dataset Files (*.csv)")
        if file[0] == "" and self.lbCheck2.isEnabled():
            return
        elif file[0] == "":
            self.ShowAlert("Lütfen bir csv seçiniz", "error")
            self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid red;}")
            return
        self.HideAlert()
        self.data = pd.read_csv(str(file[0]), index_col=0)
        self.dataarr = np.array(self.data)
        # for i, row in enumerate(self.data.columns):
        # print(str(i) + " : " + row)
        self.classCount = 0
        self.augPath = self.imPath + 'augmentated/'
        if self.cb_augmentation.currentText() == "Arttırılmış Veriseti":
            message = str(int(self.maxData.text()) * 4) + " adet veri arttırıldı."
            try:
                self.AugmentateData()
            except:
                self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid red;}")
                self.ShowAlert("Bazı fotoğraflar bulunamadı.", "error")
                return
        else:
            message = str(int(self.maxData.text())) + " adet veri seçildi."

        self.ShowAlert(message)
        try:
            self.GetXY()
        except:
            self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid red;}")
            self.ShowAlert("Bazı fotoğraflar bulunamadı.", "error")
            return
        self.lbCheck2.setEnabled(True)
        self.groupBox_2.setStyleSheet("QGroupBox { border: 1px solid green;}")
        self.groupBox_3.setEnabled(True)
        self.groupBox_3.setStyleSheet("QGroupBox { border: 1px solid blue;}")

    def AugmentateData(self):
        max = int(self.maxData.text())
        for i, row in enumerate(self.dataarr):
            fName = row[17].split('.')[0]
            ext = "." + str(row[17].split('.')[1])
            if i >= max:
                break
            if os.path.exists(self.augPath + fName + '_1' + ext) \
                    and os.path.exists(self.augPath + fName + '_2' + ext) and os.path.exists(
                self.augPath + fName + '_3' + ext) \
                    and os.path.exists(self.augPath + fName + '_4' + ext):
                continue
            try:
                img = io.imread(self.imagesPath + str(row[17]))
            except:
                max += 1
                continue
            io.imsave(self.augPath + fName + '_1' + ext, self.random_rotation(img))
            io.imsave(self.augPath + fName + '_2' + ext, self.random_noise(img))
            io.imsave(self.augPath + fName + '_3' + ext, self.horizontal_flip(img))
            io.imsave(self.augPath + fName + '_4' + ext, self.vertical_flip(img))

    def GetXY(self):
        x = []
        y = []
        yCount = 1
        max = int(self.maxData.text())
        if self.cb_augmentation.currentText() == "Arttırılmış Veriseti":
            yCount = 5
        for i, row in enumerate(self.dataarr):
            fName = row[17].split('.')[0]
            ext = "." + str(row[17].split('.')[1])
            try:
                x.append(self.GetProperImage(self.imagesPath + str(row[17])))
            except:
                max += 1
                continue
            for j in range(yCount - 1):
                x.append(self.GetProperImage(self.augPath + fName + '_' + str(j+1) + ext))
            if str(row[15][2]) not in y:
                self.classCount += 1
            for j in range(yCount):
                y.append(str(row[15][2]))
            if i >= max - 1:
                break
        x = np.array(x)
        x = np.reshape(x, (-1, 120, 120, 3))
        lb = LabelEncoder()
        y = lb.fit_transform(y)
        self.x, self.y = shuffle(x, y)

    def GetProperImage(self, path):
        img = io.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (120, 120))
        if np.max(img) > 1:
            img = img / 255.
        return img

    def TestModel(self):
        file = QFileDialog.getOpenFileName(self, 'Choose an Image to Test',
                                           str(pathlib.Path().absolute()),
                                           "Image Files (*.jpg *.jpeg *.png)")
        if file[0] == "":
            self.ShowAlert("Lütfen bir image seçiniz", "error")
            return
        self.HideAlert()
        path = str(file[0])
        fName = file[0].split("/")
        fName = fName[len(fName) - 1]
        print(fName)
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(self.lb_testImage.width(), self.lb_testImage.height())
        self.lb_testImage.setPixmap(pixmap)
        name = "model_tf"
        if self.cb_testmodel.currentIndex() == 1:
            name = "model_machine"
        X = []
        img = io.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (120, 120))
        if np.max(img) > 1:
            img = img / 255.
        X.append(img)
        X = np.array(X)
        X = np.reshape(X, (-1, 120, 120, 3))
        json_file = open('./models/' + name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("./models/" + name + ".h5")
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        result = loaded_model.predict(X)
        idx = 0
        best_value = 0
        for values in result:
            for i, value in enumerate(values):
                if value > best_value:
                    best_value = value
                    idx = i
        data = pd.read_csv("full_df.csv", index_col=0)
        dataarr = np.array(data)
        orgClass = ""
        for i, row in enumerate(dataarr):
            if row[17] == fName:
                orgClass = row[15][2]
                break
        if best_value < 0.7:
            self.lb_testResult.setText("Eşleşme bulunamadı.")
        else:
            #self.classCodes[idx]
            self.lb_testResult.setText(f"{fName}\r\nGerçek Sonuç: {self.classCodes[orgClass]}\r\nEn iyi değer: {best_value}\r\nTeşhis: {self.classNames[idx]}")

    def Split(self):
        self.groupBox_3.setStyleSheet("QGroupBox { border: 1px solid blue; }")
        self.HideAlert()
        self.tv_Xtest.clear()
        self.tb_Xtrain.clear()
        self.tv_yTest.clear()
        self.tb_yTrain.clear()
        if self.cb_splitalgo.currentText() == "Hold Out":
            self.holdOutBool = True
            percentage = self.cb_trainperc.currentText()
            self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.x, self.y,
                                                                                test_size=float(percentage),
                                                                                random_state=10)
            if self.cb_showTrainTest.isChecked():
                self.FillComboboxes([self.xTrain, self.xTest, self.yTrain, self.yTest])
        else:
            self.holdOutBool = False
            kf = KFold(n_splits=int(self.cb_k.currentText()))
            self.Kf = []
            index_uzayi = []
            for train_index, test_index in kf.split(self.x):
                arr = []
                xTrain_k, xTest_k, yTrain_k, y_test_k = [], [], [], []
                for i in train_index:
                    xTrain_k.append(self.x[i])
                    yTrain_k.append(self.y[i])
                for i in test_index:
                    xTest_k.append(self.x[i])
                    y_test_k.append(self.y[i])
                index_uzayi.append([train_index, test_index])
                arr.append(np.array(xTrain_k))
                arr.append(np.array(xTest_k))
                arr.append(np.array(yTrain_k))
                arr.append(np.array(y_test_k))
                self.Kf.append(arr)

            self.xTrain, self.xTest, self.yTrain, self.yTest, self.indexUzayi = np.array(xTrain_k), np.array(
                xTest_k), np.array(yTrain_k), np.array(y_test_k), index_uzayi

            self.cb_kfoldk.currentTextChanged.disconnect(self.ChangeKfoldK)
            self.cb_kfoldk.clear()
            for i in range(1, int(self.cb_k.currentText()) + 1):
                self.cb_kfoldk.addItem(str(i))
            self.cb_kfoldk.currentTextChanged.connect(self.ChangeKfoldK)
            self.cb_kfoldk.setCurrentIndex(int(self.cb_k.currentText()) - 1)

            self.cb_kfoldk_2.currentTextChanged.disconnect(self.ChangeKfoldK_2)
            self.cb_kfoldk_2.clear()
            for i in range(1, int(self.cb_k.currentText()) + 1):
                self.cb_kfoldk_2.addItem(str(i))
            self.cb_kfoldk_2.currentTextChanged.connect(self.ChangeKfoldK_2)

        self.y_tr_test = self.yTest
        self.y_tr_train = self.yTrain
        self.yTrain = to_categorical(self.yTrain)
        self.yTest = to_categorical(self.yTest)

        self.lbCheck3.setEnabled(True)
        self.groupBox_3.setStyleSheet("QGroupBox { border: 1px solid green; }")
        self.groupBox_4.setEnabled(True)
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid blue; }")
        self.bt_csv.setEnabled(False)
        self.maxData.setEnabled(False)
        self.cb_augmentation.setEnabled(False)
        self.bt_imgpath.setEnabled(False)

        alert = "xTrain, xTest, yTrain, yTest oluşturuldu......\r\n"
        alert += f"Sınıf Sayısı : {self.classCount}\r\n"
        alert += f"Train için {str(round((self.xTrain.shape[0] * 100) / self.y.shape[0]))}% ayrıldı, Test için {str(round((self.xTest.shape[0] * 100) / self.y.shape[0]))}% ayrıldı\r\n"
        alert += f"Train'da {str(self.xTrain.shape[0])} veri var, Test'da {str(self.xTest.shape[0])} veri var\r\n"
        self.ShowAlert(alert)

    def ChangeKfoldK(self):
        if self.cb_showTrainTest.isChecked():
            self.FillComboboxes(self.Kf[int(self.cb_kfoldk.currentText()) - 1])

    def ChangeKfoldK_2(self):
        pixmap = QPixmap(self.imPath + "graphs/confusion_fold_" + self.cb_kfoldk_2.currentText())
        pixmap = pixmap.scaled(self.lb_confusion.width(), self.lb_confusion.height())
        self.lb_confusion.setPixmap(pixmap)

    def FillComboboxes(self, arr):
        # xTrain
        a, x, y, z = arr[0].shape
        xtrain = arr[0].reshape(x * y * z, a)
        xtrain = pd.DataFrame(xtrain)
        c = len(xtrain.columns)
        r = len(xtrain.values)
        self.tb_Xtrain.setColumnCount(c)
        self.tb_Xtrain.setRowCount(r)
        for i, row in enumerate(xtrain):
            for j, cell in enumerate(xtrain.values):
                self.tb_Xtrain.setItem(j, i, QtWidgets.QTableWidgetItem(str(cell[i])))
        # xTest
        a, x, y, z = arr[1].shape
        xTest = arr[1].reshape(x * y * z, a)
        xTest = pd.DataFrame(xTest)
        c = len(xTest.columns)
        r = len(xTest.values)
        self.tv_Xtest.setColumnCount(c)
        self.tv_Xtest.setRowCount(r)
        for i, row in enumerate(xTest):
            for j, cell in enumerate(xTest.values):
                self.tv_Xtest.setItem(j, i, QtWidgets.QTableWidgetItem(str(cell[i])))
        # yTrain
        yTrain = pd.DataFrame(arr[2])
        c = len(yTrain.columns)
        r = len(yTrain.values)
        self.tb_yTrain.setColumnCount(c)
        self.tb_yTrain.setRowCount(r)
        for i, row in enumerate(yTrain):
            for j, cell in enumerate(yTrain.values):
                self.tb_yTrain.setItem(j, i, QtWidgets.QTableWidgetItem(str(cell[i])))
        # yTest
        yTest = pd.DataFrame(arr[3])
        c = len(yTest.columns)
        r = len(yTest.values)
        self.tv_yTest.setColumnCount(c)
        self.tv_yTest.setRowCount(r)
        for i, row in enumerate(yTest):
            for j, cell in enumerate(yTest.values):
                self.tv_yTest.setItem(j, i, QtWidgets.QTableWidgetItem(str(cell[i])))

    def Train(self):
        self.TransferLearning = False
        self.ProceedTransferLearning()

    def TransferLearningClick(self):
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid blue; }")
        self.TransferLearning = True
        self.TransferLearningDone = False
        self.ProceedTransferLearning()

    def ProceedTransferLearning(self):
        if not self.TransferLearningDone:
            print("transfer learning yapıldı..")
            modelName = self.cb_transferalgo.currentText()
            if modelName == "AlexNet":
                self.model = self.AlexNet()
            elif modelName == "VGGNet":
                self.model = self.VGGNet()

            lrate = 0.001
            decay = lrate / 100
            sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
            self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            self.model.fit(self.xTrain, self.yTrain, validation_split=0.5, epochs=int(self.tb_epoch.text()),
                                     batch_size=16)

        if not self.TransferLearning:
            layer_name = 'dense1'
            self.FC_layer_model = Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
            features = np.zeros(shape=(self.xTrain.shape[0], 4096))
            for i in range(len(self.xTrain)):
                img = np.expand_dims(self.xTrain[i], axis=0)
                FC_output = self.FC_layer_model.predict(img)
                features[i] = FC_output
            feature_col = []
            for i in range(4096):
                feature_col.append("f_" + str(i))
                i += 1
            train_features = pd.DataFrame(data=features, columns=feature_col)

            machineAlgo = self.cb_machinealgo.currentText()
            if machineAlgo == 'KNN':
                self.model_machine = OneVsRestClassifier(KNeighborsClassifier(5, weights='distance'))
                self.model_machine.fit(train_features, self.y_tr_train)
            elif machineAlgo == 'Random Forest':
                self.model_machine = OneVsRestClassifier(RandomForestClassifier(random_state=0))
                self.model_machine.fit(train_features, self.y_tr_train)
            elif machineAlgo == 'LR':
                self.model_machine = OneVsRestClassifier(LogisticRegression(random_state=0))
                self.model_machine.fit(train_features, self.y_tr_train)
            elif machineAlgo == 'DT':
                self.model_machine = OneVsRestClassifier(DecisionTreeClassifier(random_state=0))
                self.model_machine.fit(train_features, self.y_tr_train)

        message = "Transfer eğitimi tamamlandı."
        if self.TransferLearning:
            self.PredictTF()  # sadece transfer ise bu predict.
        else:
            self.Predict()
            message = "Makine eğitimi tamamlandı."
            self.lbCheck5.setEnabled(True)

        self.TransferLearningDone = True
        self.ShowAlert(message)
        self.bt_csv.setEnabled(False)
        self.bt_imgpath.setEnabled(False)
        self.lbCheck4.setEnabled(True)
        self.bt_saveModel.setEnabled(True)
        self.groupBox_4.setStyleSheet("QGroupBox { border: 1px solid green; }")
        self.groupBox_5.setEnabled(True)
        self.groupBox_5.setStyleSheet("QGroupBox { border: 1px solid blue; }")


    def Predict(self):
        if self.holdOutBool:
            features = np.zeros(shape=(self.xTest.shape[0], 4096))
            for i in range(len(self.xTest)):
                img = np.expand_dims(self.xTest[i], axis=0)
                FC_output = self.FC_layer_model.predict(img)
                features[i] = FC_output
            feature_col = []
            for j in range(4096):
                feature_col.append("f_" + str(j))
                j += 1
            test_features = pd.DataFrame(data=features, columns=feature_col)
            predicts = self.model_machine.predict(test_features)
            pred_prob = self.model_machine.predict_proba(test_features)
            predict = pred_prob, predicts
            self.ConfusionMatrix(self.y_tr_test, predicts)
        else:
            cms = []
            for idx, row in enumerate(self.indexUzayi):
                for j, col in enumerate(row):
                    xTest = []
                    yTest = []
                    if j == 1:
                        for k in col:
                            xTest.append(self.x[k])
                            yTest.append(self.y[k])
                        xTest = np.array(xTest)
                        yTest = np.array(yTest)
                        features = np.zeros(shape=(xTest.shape[0], 4096))
                        for l in range(len(xTest)):
                            img = np.expand_dims(xTest[l], axis=0)
                            FC_output = self.FC_layer_model.predict(img)
                            features[l] = FC_output
                        feature_col = []
                        for t in range(4096):
                            feature_col.append("f_" + str(t))
                            t += 1
                        test_features = pd.DataFrame(data=features, columns=feature_col)
                        predicts = self.model_machine.predict(test_features)
                        pred_prob = self.model_machine.predict_proba(test_features)
                        predict = pred_prob, predicts
                        cms.append(self.ConfusionMatrix(yTest, predicts, idx))
            self.OverlappedMatrix(cms)

        self.RocCurve(self.y_tr_test, pred_prob)

    def PredictTF(self):
        y = to_categorical(self.y)
        if self.holdOutBool:
            result = self.model.predict(self.xTest)
            yPred = np.argmax(result, axis=1)
            self.ConfusionMatrix(self.y_tr_test, yPred)
        else:
            overlap = []
            for idx, row in enumerate(self.indexUzayi):
                for j, col in enumerate(row):
                    xTest = []
                    yTest = []
                    yTrue = []
                    if j == 1:
                        for k in col:
                            xTest.append(self.x[k])
                            yTest.append(y[k])
                            yTrue.append(self.y[k])
                        xTest = np.array(xTest)
                        yTest = np.array(yTest)
                        yTrue = np.array(yTrue)
                        result = self.model.predict(self.xTest)
                        yPred = []
                        for i in result:
                            temp = []
                            for j in i:
                                if j < 0.5:
                                    temp.append(0)
                                elif j >= 0.5:
                                    temp.append(1)
                            a = False
                            for v, k in enumerate(temp):
                                if k == 1:
                                    a = True
                                    yPred.append(v)
                            if not a:
                                yPred.append(0)
                        overlap.append(self.ConfusionMatrix(np.argmax(yTest, axis=1), yTrue, idx))
                self.OverlappedMatrix(overlap)
        # self.cb_kfoldk_2.currentTextChanged.connect(self.ChangeKfoldK_2)
        self.cb_kfoldk_2.setCurrentIndex(int(self.cb_k.currentText()) - 1)

    def VGGNet(self):
        vgg = VGG16(weights='imagenet', input_shape=(120, 120, 3), include_top=False)
        model = Sequential()
        for layer in vgg.layers:
            model.add(layer)
        for layer in model.layers:
            layer.trainable = False
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', kernel_constraint=maxnorm(3), name='dense1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', kernel_constraint=maxnorm(3), name='dense2'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classCount, activation='softmax', name='outPut'))
        model.summary()
        return model

    def AlexNet(self):
        shape = (120, 120, 3)
        X_input = Input(shape)
        X = Conv2D(96, (11, 11), strides=4, name="conv0")(X_input)
        X = BatchNormalization(axis=3, name="bn0")(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=2, name='max0')(X)
        X = Conv2D(256, (5, 5), padding='same', name='conv1')(X)
        X = BatchNormalization(axis=3, name='bn1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=2, name='max1')(X)
        X = Conv2D(384, (3, 3), padding='same', name='conv2')(X)
        X = BatchNormalization(axis=3, name='bn2')(X)
        X = Activation('relu')(X)
        X = Conv2D(384, (3, 3), padding='same', name='conv3')(X)
        X = BatchNormalization(axis=3, name='bn3')(X)
        X = Activation('relu')(X)
        X = Conv2D(256, (3, 3), padding='same', name='conv4')(X)
        X = BatchNormalization(axis=3, name='bn4')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=2, name='max2')(X)
        X = Flatten()(X)
        X = Dense(4096, activation='relu', name="dense1")(X)
        X = Dense(4096, activation='relu', name='fc1')(X)
        X = Dense(self.classCount, activation='softmax', name='fc2')(X)
        model = Model(inputs=X_input, outputs=X, name='AlexNet')
        model.summary()
        return model

    def RocCurve(self, yTest, pred_prob):
        plt.clf()
        fpr = {}
        tpr = {}
        thresh = {}
        colors = ['orange', 'green', 'blue']
        try:
            if self.TransferLearning:
                algo = self.cb_transferalgo.currentText()
            else:
                algo = self.cb_machinealgo.currentText()
            for i in range(pred_prob.shape[1]):
                fpr[i], tpr[i], thresh[i] = roc_curve(yTest, pred_prob[:, i], pos_label=i)
            k = 2
            for j in range(pred_prob.shape[1]):
                if k == 3:
                    k = 0
                plt.plot(fpr[j], tpr[j], linestyle='--', color=str(colors[k]), label='ROC curve of class {0}'''.format(j+1))
                k = k + 1
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(algo + ' ROC Curve (' + datetime.now().strftime("%m/%d/%Y") + ')')
            plt.legend(loc='best')
            plt.savefig(self.imPath + 'graphs/roc.png', dpi=100)
        except:
            print("Bir hata oluştu")
        pixmap = QPixmap(self.imPath + 'graphs/roc.png')
        pixmap = pixmap.scaled(self.lb_roc.width(), self.lb_roc.height())
        self.lb_roc.setPixmap(pixmap)

    def ConfusionMatrix(self, yTrue, yPred, fold=0):
        cm = confusion_matrix(yTrue, yPred)
        cm_data = pd.DataFrame(cm)
        plt.clf()
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_data, annot=True, fmt="d")
        if self.TransferLearning:
            algo = self.cb_transferalgo.currentText()
        else:
            algo = self.cb_machinealgo.currentText()
        if self.holdOutBool:
            title = algo + " Hold Out Conf. Mat."
            name = "graphs/confusion_hold-out.png"
        else:
            title = algo + " K-Fold Conf. Mat. - " + str(fold + 1)
            name = "graphs/confusion_fold_" + str(fold + 1) + ".png"
        title += ' (' + datetime.now().strftime("%m/%d/%Y") + ')'
        plt.title(title)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig(self.imPath + name)
        pixmap = QPixmap(self.imPath + name)
        pixmap = pixmap.scaled(self.lb_confusion.width(), self.lb_confusion.height())
        self.lb_confusion.setPixmap(pixmap)
        return cm

    def OverlappedMatrix(self, cms):
        cms = np.array(cms)
        overlapped = np.zeros(cms[0].shape)
        for c in cms:
            if overlapped.shape == c.shape:
                overlapped = np.add(overlapped, c)
        overlapped = overlapped.astype(int)
        cm_data = pd.DataFrame(overlapped)
        plt.clf()
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_data, annot=True, fmt="d")
        name = "graphs/overlapped.png"
        if self.TransferLearning:
            algo = self.cb_transferalgo.currentText()
        else:
            algo = self.cb_machinealgo.currentText()
        plt.title(algo + " Overlapped Matrix (" + datetime.now().strftime("%m/%d/%Y") + ")")
        plt.ylabel('y')
        plt.xlabel('x')
        plt.savefig(self.imPath + name)
        pixmap = QPixmap(self.imPath + name)
        pixmap = pixmap.scaled(self.lb_overlapped.width(), self.lb_overlapped.height())
        self.lb_overlapped.setPixmap(pixmap)

    def SaveCurrentModel(self):
        if self.TransferLearning:
            self.SaveModel(self.model, "model_tf")
        else:
            self.SaveModel(self.model, "model_machine")
        self.ShowAlert("Model başarıyla kaydedildi.")

    def SaveModel(self, model, name):
        # modeli daha sonra test için kullanmak üzere .json olarak kaydeder.
        try:
            model_json = model.to_json()
            with open("./models/" + name + ".json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights("./models/" + name + ".h5")
        except:
            return

    def random_rotation(self, img):
        # -25 ile 25 derece arasında random bir döndürme uygular
        random_degree = random.uniform(-25, 25)
        return skimage.transform.rotate(img, random_degree)

    def random_noise(self, img):
        # fotoğrafa random gürültü ekler
        return skimage.util.random_noise(img)

    def horizontal_flip(self, img):
        # fotoğrafı yatayda ters çevirir
        return img[:, ::-1]

    def vertical_flip(self, img):
        # fotoğrafı dikeyde ters çevirir
        return img[::-1, :]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = window()
    window.show()
    sys.exit(app.exec())
