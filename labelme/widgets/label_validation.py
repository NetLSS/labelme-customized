import cv2
import os
import glob
import random
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPalette, QImage, qRgb, QPixmap
from labelme.cli.validation_json import *

from_class = uic.loadUiType("./widgets/label_validation.ui")[0]


class LabelValidationDialog(QDialog, from_class):
    def __init__(self):

        super().__init__()
        self.setupUi(self)

        self.gray_color_table = [qRgb(i, i, i) for i in range(256)]

        self.pushButton_openA.clicked.connect(self.onButtonClickOpenA)
        self.pushButton_openB.clicked.connect(self.onButtonClickOpenB)
        self.pushButton_process.clicked.connect(self.onButtonClickProcess)

        self.progressBar.setValue(0)

        # self.label_imageA.setBackgroundRole(QPalette.Base)
        # self.label_imageA.setScaledContents(True)


    def onButtonClickOpenA(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if path:
            self.lineEdit_folderA.setText(path)

    def onButtonClickOpenB(self):
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if path:
            self.lineEdit_folderB.setText(path)

    def onButtonClickProcess(self):
        true_json_folder_path = self.lineEdit_folderA.text()
        target_json_folder_path = self.lineEdit_folderB.text()

        result_path = osp.join(osp.dirname(true_json_folder_path), "result")
        if not osp.exists(result_path):
            os.mkdir(result_path)

        true_json_list = glob.glob(os.path.join(true_json_folder_path, "*.json"))
        # target_json_list = glob.glob(os.path.join(target_json_folder_path, "*.json"))

        self.progressBar.setValue(0)
        self.progressBar.setMaximum(len(true_json_list))

        not_matched_files = []
        for i, true_json in enumerate(true_json_list):
            target_json = osp.join(target_json_folder_path, osp.basename(true_json))
            if not osp.exists(target_json):
                result_list.append(
                    [len(result_list), osp.basename(true_json), "None", "None", "None", "None", "None", "None", "None"])
                not_matched_files.append(target_json)
            else:
                validate_json_file(true_json, target_json, result_path)
            self.progressBar.setValue(i+1)


        save_result_csv(osp.join(result_path, "result_total.csv"))

    def tpQImage(self, im, copy=False):
        if im is None:
            return QImage()

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(self.gray_color_table)
                return qim.copy() if copy else qim
            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim

    def openImage(self, image=None, fileName=None):
        if image == None:
            image = QImage(fileName)
        if image.isNull():
            QMessageBox.information(self, "Image Viewer",
                                    "Cannot load %s." % fileName)
            return

        self.imageLabel.setPixmap(QPixmap.fromImage(image))
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()
        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()
    #     self.pushButton_open_logo.clicked.connect(self.buttonLogoOpen)
    #     self.pushButton_open_back.clicked.connect(self.buttonBackOpen)
    #     self.pushButton_open_save.clicked.connect(self.buttonSaveOpen)
    #     self.pushButton_process.clicked.connect(self.buttonProcess)
    #
    # def buttonLogoOpen(self):
    #     path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    #     self.lineEdit_logo.setText(path)
    #
    # def buttonBackOpen(self):
    #     path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    #     self.lineEdit_back.setText(path)
    #
    # def buttonSaveOpen(self):
    #     path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    #     self.lineEdit_save.setText(path)
    #
    # def buttonProcess(self):
    #     imgExt = self.comboBox_logo.currentText()  # '.jpg'  # ✅
    #     back_imgExt = self.comboBox_back.currentText()  # '.jpg'  # ✅
    #     save_imgExt = self.comboBox_save.currentText()  # '.jpg'  # ✅
    #     savePath = self.lineEdit_save.text() + '/'  # "C:/Users/sangsu lee/Desktop/test/"  # ✅
    #     rplogoPath = self.lineEdit_logo.text()
    #
    #     save_file_name = os.path.basename(rplogoPath).split('.')[0]  # 확장자 없는 파일명만 추출
    #
    #     logoImagesPath = self.lineEdit_logo.text() + '/*'  # 'D:/2020/DS/Project/2020-05-22-yolo-v4-darknet-master/00.train-set/00.train-set-final/3.broken-crop/*'  # ✅
    #     file_list = glob.glob(logoImagesPath)
    #     logo_image_list = [file for file in file_list if file.endswith(imgExt)]
    #     print(logo_image_list)
    #
    #     backgroundPath = self.lineEdit_back.text() + '/*'  # 'D:/2020/DS/Project/2020-05-22-yolo-v4-darknet-master/00.train-set/00.train-set-final/_9.nowire_background/*'  # ✅
    #     background_image_filelist = glob.glob(backgroundPath)
    #     background_image_list = [
    #         file for file in background_image_filelist if file.endswith(back_imgExt)]
    #
    #     # yolo label setting
    #     classes = 3
    #
    #     for logoImg in logo_image_list:
    #         backgroundPath = background_image_list[random.randint(
    #             0, len(background_image_list) - 1)]
    #
    #         backgroundImg = cv2.imread(backgroundPath, 1)
    #         rplogoImg = cv2.imread(logoImg, 1)
    #         print(logoImg)
    #
    #         basename_splited_comma = os.path.basename(logoImg).split('.')
    #         save_file_name = basename_splited_comma[len(basename_splited_comma) - 2]
    #
    #         # Image info
    #         print('===== Original Image Info =====')
    #         print('Background Image : ', backgroundImg.shape)
    #         print('CP Image  : ', rplogoImg.shape)
    #
    #         back_size_y, back_size_x, back_channel = backgroundImg.shape  # 이미지 크기 및 채널 받아오기
    #         cp_size_y, cp_size_x, cp_channel = rplogoImg.shape
    #
    #         # Image Resize
    #         rplogoImgDownscale = rplogoImg  # cv2.resize(rplogoImg, None,
    #         # fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    #
    #         print('===== Resized CP Logo Image Info =====')
    #         print('CP Image  : ', rplogoImgDownscale.shape)
    #
    #         # Image Addition
    #         x_offset = y_offset = 100
    #
    #         # 합성할 offset 설정. (정가운데로 설정)
    #         y_offset = int((backgroundImg.shape[0] / 2) - (rplogoImgDownscale.shape[0] / 2))
    #         x_offset = int((backgroundImg.shape[1] / 2) - (rplogoImgDownscale.shape[1] / 2))
    #
    #         # 이미지 합성 (덮어쓰기)
    #         backgroundImg[y_offset:y_offset + rplogoImgDownscale.shape[0],
    #         x_offset:x_offset + rplogoImgDownscale.shape[1]] = rplogoImgDownscale
    #
    #         # Yolo Labeling
    #         # <object-class> <x_center> <y_center> <width> <height>
    #         print(classes, " ", 0.5, " ", 0.5, " ",
    #               cp_size_x / back_size_x, cp_size_y / back_size_y)
    #         # examples)
    #         # 0 0.5 0.5 1.0 1.0
    #         # 0 0.287891 0.602604 0.307031 0.048958
    #         # 0   0.5   0.5   0.328125 0.03333333333333333
    #
    #         # Result
    #         # cv2.imshow('Background with CP Logo', backgroundImg)
    #         cv2.moveWindow('Background with CP Logo', 50, 50)
    #         # image save
    #         cv2.imwrite(savePath + save_file_name + save_imgExt, backgroundImg)
    #         print(savePath + save_file_name + save_imgExt)
    #         # cv2.waitKey(300)
    #         # cv2.destroyAllWindows()
    #
    #         # label txt file save
    #         f = open(savePath + save_file_name + ".txt", "w")
    #         f.write(str(classes) + " 0.5 0.5 " + str(cp_size_x /
    #                                                  back_size_x) + " " + str(cp_size_y / back_size_y))
    #         f.close()

        # cv2.waitKey(0)

        # cv2.destroyAllWindows()

        # 버튼에 기능을 연결하는 코드
        # self.pushButton_makeCommand.clicked.connect(self.button1Function)
        # self.pushButton_clear.clicked.connect(self.button_clear_Function)


#     def button1Function(self):
#         darknet_path = self.lineEdit_darknet.text().replace('\\','/')
#         data_path = self.lineEdit_data.text().replace('\\','/')
#         cfg_path = self.lineEdit_cfg.text().replace('\\','/')
#         conv_path = self.lineEdit_conv.text().replace('\\','/')
#         self.plainTextEdit.setPlainText(darknet_path + ' detector' + ' train '+ data_path+' '+cfg_path+' '+conv_path)

#     def button_clear_Function(self):
#         self.lineEdit_darknet.clear()
#         self.lineEdit_data.clear()
#         self.lineEdit_cfg.clear()
#         self.lineEdit_conv.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    MyWindow = LabelValidationWindow()
    MyWindow.show()
    app.exec_()

# path setting
# os.path.basename(filename) - 파일명만 추출
# imgExt = '.jpg'  # ✅
# back_imgExt = '.jpg'  # ✅
# save_imgExt = '.jpg'  # ✅
# savePath = "C:/Users/sangsu lee/Desktop/test/"  # ✅

# backgroundPath = "D:/2020/DS/Project/2020-05-22-yolo-v4-darknet-master/00.train-set/00.train-set-final/_9.nowire_background/72.jpg"
# rplogoPath = "D:/2020/DS/Project/2020-05-22-yolo-v4-darknet-master/00.train-set/00.train-set-final/6.doubleBond-crop/edtd_0157-test-03138-1crop.jpg"
# backgroundImg = cv2.imread(backgroundPath, 1)
# rplogoImg = cv2.imread(rplogoPath, 1)

# save_file_name = os.path.basename(rplogoPath).split('.')[0]  # 확장자 없는 파일명만 추출

# logoImagesPath = 'D:/2020/DS/Project/2020-05-22-yolo-v4-darknet-master/00.train-set/00.train-set-final/3.broken-crop/*'  # ✅
# file_list = glob.glob(logoImagesPath)
# logo_image_list = [file for file in file_list if file.endswith(imgExt)]
# print(logo_image_list)


# backgroundPath = 'D:/2020/DS/Project/2020-05-22-yolo-v4-darknet-master/00.train-set/00.train-set-final/_9.nowire_background/*'  # ✅
# background_image_filelist = glob.glob(backgroundPath)
# background_image_list = [
#     file for file in background_image_filelist if file.endswith(back_imgExt)]


# # yolo label setting
# classes = 3


# for logoImg in logo_image_list:

#     backgroundPath = background_image_list[random.randint(
#         0, len(background_image_list)-1)]

#     backgroundImg = cv2.imread(backgroundPath, 1)
#     rplogoImg = cv2.imread(logoImg, 1)
#     print(logoImg)

#     basename_splited_comma = os.path.basename(logoImg).split('.')
#     save_file_name = basename_splited_comma[len(basename_splited_comma)-2]

#     # Image info
#     print('===== Original Image Info =====')
#     print('Background Image : ', backgroundImg.shape)
#     print('CP Image  : ', rplogoImg.shape)

#     back_size_y, back_size_x, back_channel = backgroundImg.shape  # 이미지 크기 및 채널 받아오기
#     cp_size_y, cp_size_x, cp_channel = rplogoImg.shape

#     # Image Resize
#     rplogoImgDownscale = rplogoImg  # cv2.resize(rplogoImg, None,
#     # fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

#     print('===== Resized CP Logo Image Info =====')
#     print('CP Image  : ', rplogoImgDownscale.shape)

#     # Image Addition
#     x_offset = y_offset = 100

#     # 합성할 offset 설정. (정가운데로 설정)
#     y_offset = int((backgroundImg.shape[0]/2)-(rplogoImgDownscale.shape[0]/2))
#     x_offset = int((backgroundImg.shape[1]/2)-(rplogoImgDownscale.shape[1]/2))

#     # 이미지 합성 (덮어쓰기)
#     backgroundImg[y_offset:y_offset + rplogoImgDownscale.shape[0],
#                   x_offset:x_offset + rplogoImgDownscale.shape[1]] = rplogoImgDownscale

#     # Yolo Labeling
#     # <object-class> <x_center> <y_center> <width> <height>
#     print(classes, " ", 0.5, " ", 0.5, " ",
#           cp_size_x/back_size_x, cp_size_y/back_size_y)
#     # examples)
#     # 0 0.5 0.5 1.0 1.0
#     # 0 0.287891 0.602604 0.307031 0.048958
#     # 0   0.5   0.5   0.328125 0.03333333333333333

#     # Result
#     #cv2.imshow('Background with CP Logo', backgroundImg)
#     cv2.moveWindow('Background with CP Logo', 50, 50)
#     # image save
#     cv2.imwrite(savePath+save_file_name + save_imgExt, backgroundImg)
#     print(savePath+save_file_name + save_imgExt)
#     # cv2.waitKey(300)
#     # cv2.destroyAllWindows()

#     # label txt file save
#     f = open(savePath+save_file_name+".txt", "w")
#     f.write(str(classes)+" 0.5 0.5 "+str(cp_size_x /
#                                          back_size_x)+" "+str(cp_size_y/back_size_y))
#     f.close()


# # cv2.waitKey(0)

# # cv2.destroyAllWindows()
