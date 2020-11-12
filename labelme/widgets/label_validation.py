import cv2
import os
import os.path as osp
import glob
import random
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPalette, QImage, qRgb, QPixmap
#from labelme.cli.validation_json import *
import subprocess

import clipboard
import argparse
import json
import numpy as np
from enum import Enum
import csv
from labelme.label_file import LabelFile
from labelme.logger import logger
from labelme import utils

import imgviz
import matplotlib.pyplot as plt


from_class = uic.loadUiType("./widgets/label_validation.ui")[0]

ROUND_NUMBER = 4


class ResultEnum(Enum):
    INDEX = 0
    FILE_A = 1
    FILE_B = 2
    LOWEST_CLASS = 3
    CLASS_NAME = 4
    ACCURACY = 5
    IOU = 6
    AVG_ACC = 7
    AVG_IOU = 8

result_list = [["index", "file A", "file B", "lowest class", "class name", "accuracy", "iou", "avg acc", "avg iou"],]

class ResultData:
    """
    json 파일1
    json 파일2
    클래스 이름 list
    accuracy list
    iou list
    lowest accuracy value
    lowest iou value
    lowest accuracy index
    lowest iou index

    """
    def __init__(self, json_true: str, json_target: str, class_names: list, accuracy_list: list, iou_list: list):
        self.json_true = json_true
        self.json_target = json_target
        self.class_names = class_names
        self.accuracy_list = accuracy_list
        self.iou_list = iou_list

        self.lowest_acc = min(self.accuracy_list)
        self.lowest_iou = min(self.iou_list)

        self.lowest_acc_i = self.accuracy_list.index(self.lowest_acc)
        self.lowest_iou_i = self.iou_list.index(self.lowest_iou)

        self.vis_img_true = None
        self.vis_img_target = None

        self.vis_Qimg_true = None
        self.vis_Qimg_target = None


class LabelValidationDialog(QDialog, from_class):
    # filteredResultDataList: list[ResultData]

    def __init__(self):

        super().__init__()
        # region pyQT UI
        self.setupUi(self)

        self.gray_color_table = [qRgb(i, i, i) for i in range(256)]

        self.pushButton_openA.clicked.connect(self.onButtonClickOpenA)
        self.pushButton_openB.clicked.connect(self.onButtonClickOpenB)
        self.pushButton_process.clicked.connect(self.onButtonClickProcess)
        self.pushButton_left.clicked.connect(self.onButtonClickLeft)
        self.pushButton_right.clicked.connect(self.onButtonClickRight)
        self.pushButton_filter_apply.clicked.connect(self.onButtonClickFilterApply)

        self.progressBar.setValue(0)

        self.lineEdit_folderA.setText("D:/2020/DS/Project/2020-11-02-labelme/labelme-master/labelme/cli/validataion_example/true_label")
        self.lineEdit_folderB.setText("D:/2020/DS/Project/2020-11-02-labelme/labelme-master/labelme/cli/validataion_example/user_label")

        self.lineEdit_iouThreshold.editingFinished.connect(self.onEditionFinishedIouThreshold)


        # endregion

        self.resultDataList = []
        self.filteredResultDataList = []
        self.current_image_index = -1

    def onButtonClickFilterApply(self):
        self.updateFilteredResultList()
        QMessageBox.information(self, "information", "Applied.")


    def onButtonClickRight(self):

        if len(self.filteredResultDataList) <= 0:
            return
        self.label_image_index.setText("opening...")
        self.current_image_index += 1

        if self.current_image_index > len(self.filteredResultDataList):
            self.current_image_index = len(self.filteredResultDataList)

        self.updateImageLabel()

    def onButtonClickLeft(self):
        if len(self.filteredResultDataList) <= 0:
            return
        self.label_image_index.setText("opening...")

        self.current_image_index -= 1

        if self.current_image_index <= 0:
            self.current_image_index = 1

        self.updateImageLabel()

    def onEditionFinishedIouThreshold(self):
        txt = self.lineEdit_iouThreshold.text()
        if not txt.isdigit():
            self.lineEdit_iouThreshold.setText("0.5")

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
        iou_treshold_value = float(self.lineEdit_iouThreshold.text())

        self.resultDataList.clear()

        result_path = osp.join(osp.dirname(true_json_folder_path), "result")
        if not osp.exists(result_path):
            os.mkdir(result_path)

        true_json_list = glob.glob(os.path.join(true_json_folder_path, "*.json"))
        # target_json_list = glob.glob(os.path.join(target_json_folder_path, "*.json"))

        self.progressBar.setValue(0)
        self.progressBar.setMaximum(len(true_json_list))
        self.label_total_index.setText(str(len(true_json_list)))

        not_matched_files = []
        for i, true_json in enumerate(true_json_list):
            target_json = osp.join(target_json_folder_path, osp.basename(true_json))
            if not osp.exists(target_json):
                result_list.append(
                    [len(result_list), osp.basename(true_json), "None", "None", "None", "None", "None", "None", "None"])
                not_matched_files.append(target_json)
            else:
                self.validate_json_file(true_json, target_json, result_path, iou_treshold_value)

                data_true = json.load(open(true_json))
                data_target = json.load(open(target_json))

                img_true = self.image_data_load(data_true)
                img_target = self.image_data_load(data_target)

                img_true = np.array(img_true)
                img_target = np.array(img_target)

                # Qimage_true = self.tpQImage(img_true)#QImage(img_true.data, h, w, QImage.Format_Indexed8)
                # Qimage_target = self.tpQImage(img_target)
                #
                # self.label_imageA.setPixmap(QPixmap.fromImage(Qimage_true))
                # self.label_imageB.setPixmap(QPixmap.fromImage(Qimage_target))

                # self.label_imageA.adjustSize()

            self.progressBar.setValue(i+1)

        self.save_result_csv(osp.join(result_path, "result_total.csv"))
        self.updateFilteredResultList()

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

    def image_data_load(self, data_true):
        imageData = data_true.get("imageData")
        img = utils.img_b64_to_arr(imageData)
        return img

    def lbl_arr_load(self, data_true, img_true):
        label_name_to_value = {"_background_": 0}
        for shape in sorted(data_true["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img_true.shape, data_true["shapes"], label_name_to_value
        )
        return lbl

    def load_label_names(self, data, img):
        label_name_to_value = {"_background_": 0}
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, data["shapes"], label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name

        return label_names

    def save_result_csv(self, save_full_path, add_mode=False):
        open_mode = 'a' if add_mode else 'w'

        try:
            with open(save_full_path, open_mode, encoding='utf-8', newline='') as writer_csv:
                writer = csv.writer(writer_csv, delimiter=",")
                for row in result_list:
                    writer.writerow(row)

        except PermissionError:
            print("[PermissionError] file save fail. if file opened please close file and retry.")


    def make_label_visualize_image(self, json_file):
        label_file = LabelFile(json_file)
        img = utils.img_data_to_arr(label_file.imageData)

        label_name_to_value = {"_background_": 0}
        for shape in sorted(label_file.shapes, key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl, _ = utils.shapes_to_label(
            img.shape, label_file.shapes, label_name_to_value
        )

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        lbl_viz = imgviz.label2rgb(
            label=lbl,
            img=imgviz.asgray(img),
            label_names=label_names,
            font_size=30,
            loc="rb",
        )

        return lbl_viz

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)
        ax2 = fig.add_subplot(111)
        ax2.imshow(lbl_viz)

        fig.canvas.draw()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # plt.subplot(121)
        # plt.imshow(img)
        # plt.subplot(122)
        # plt.imshow(lbl_viz)
        # plt.show()
        #
        # plt.draw()

        return data


    def updateImageLabel(self):
        if self.current_image_index <= 0:
            return

        arr_index = self.current_image_index - 1

        if self.filteredResultDataList[arr_index].vis_img_true:
            vis_img_true = self.filteredResultDataList[arr_index].vis_img_true
        else:
            vis_img_true = self.make_label_visualize_image(self.filteredResultDataList[arr_index].json_true)
        if self.filteredResultDataList[arr_index].vis_img_target:
            vis_img_target = self.filteredResultDataList[arr_index].vis_img_target
        else:
            vis_img_target = self.make_label_visualize_image(self.filteredResultDataList[arr_index].json_target)

        if self.filteredResultDataList[arr_index].vis_Qimg_true:
            Qimage_true = self.filteredResultDataList[arr_index].vis_Qimg_true
        else:
            Qimage_true = self.tpQImage(vis_img_true)
        if self.filteredResultDataList[arr_index].vis_Qimg_target:
            Qimage_target = self.filteredResultDataList[arr_index].vis_Qimg_target
        else:
            Qimage_target = self.tpQImage(vis_img_target)

        # TODO: speed issue
        self.label_imageA.setPixmap(QPixmap.fromImage(Qimage_true))
        self.label_imageB.setPixmap(QPixmap.fromImage(Qimage_target))

        filter_mode = self.comboBox_mode.currentText()
        if filter_mode == "iou":
            self.label_class_name.setText(str(self.filteredResultDataList[arr_index].class_names[self.filteredResultDataList[arr_index].lowest_iou_i]))
            self.label_class_accuracy.setText(str(self.filteredResultDataList[arr_index].accuracy_list[self.filteredResultDataList[arr_index].lowest_iou_i]))
            self.label_class_iou.setText(str(self.filteredResultDataList[arr_index].lowest_iou))
        elif filter_mode == "accuracy":
            self.label_class_name.setText(
                str(self.filteredResultDataList[arr_index].class_names[self.filteredResultDataList[arr_index].lowest_acc_i]))
            self.label_class_accuracy.setText(str(self.filteredResultDataList[arr_index].lowest_acc))
            self.label_class_iou.setText(str(self.filteredResultDataList[arr_index].iou_list[self.filteredResultDataList[arr_index].lowest_acc_i]))


        self.label_imageA_name.setText(osp.basename(self.filteredResultDataList[arr_index].json_true))
        self.label_imageB_name.setText(osp.basename(self.filteredResultDataList[arr_index].json_target))

        self.label_image_index.setText(f"{self.current_image_index}/{len(self.filteredResultDataList)}")


    def updateFilteredResultList(self):
        filter_mode = self.comboBox_mode.currentText()
        filter_threshold = float(self.lineEdit_threshold.text())
        self.filteredResultDataList.clear()

        for result_data in self.resultDataList:
            #result_data = ResultData(result_data)
            if filter_mode == "iou":
                if result_data.lowest_iou <= filter_threshold:
                    self.filteredResultDataList.append(result_data)
            elif filter_mode == "accuracy":
                if result_data.lowest_acc <= filter_threshold:
                    self.filteredResultDataList.append(result_data)

        if len(self.filteredResultDataList) <= 0:
            self.current_image_index = -1
            QLabel(self.label_image_index).setText("0/0")
            QPushButton(self.pushButton_left).setEnabled(False)
            QPushButton(self.pushButton_right).setEnabled(False)
            return

        self.current_image_index = 1
        self.label_image_index.setText(f"{self.current_image_index}/{len(self.filteredResultDataList)}")
        self.updateImageLabel()

        # self.filteredResultDataList

        pass

    def validate_json_file(self, true_json, target_json, out_full_path=None, threshold=0.5, show_image=False):
        json_file_true = true_json
        json_file_target = target_json

        # region 저장 경로 설정/생성
        if out_full_path is None:
            out_dir = osp.basename(json_file_true).replace(".", "_")
            out_dir = osp.join(osp.dirname(json_file_true), out_dir)
        else:
            out_dir = out_full_path
        if not osp.exists(out_dir):
            os.mkdir(out_dir)
        # endregion

        data_true = json.load(open(json_file_true))
        data_target = json.load(open(json_file_target))

        img_true = self.image_data_load(data_true)
        img_target = self.image_data_load(data_target)

        label_names_true = self.load_label_names(data_true, img_true)
        label_names_target = self.load_label_names(data_target, img_target)

        lbl_true = self.lbl_arr_load(data_true, img_true)
        lbl_target = self.lbl_arr_load(data_target, img_target)

        lbl_true = np.array(lbl_true)
        lbl_target = np.array(lbl_target)

        if show_image:
            Qimage_true = self.tpQImage(img_true)  # QImage(img_true.data, h, w, QImage.Format_Indexed8)
            Qimage_target = self.tpQImage(img_target)

            self.label_imageA.setPixmap(QPixmap.fromImage(Qimage_true))
            self.label_imageB.setPixmap(QPixmap.fromImage(Qimage_target))

        # print(f"lbl_true {lbl_true}")
        # print(f"lbl_target {lbl_target}")

        # print(f"np.max(lbl_true) {np.max(lbl_true)}")
        # print(f"np.max(lbl_target) {np.max(lbl_target)}")
        max_label_number = max(np.max(lbl_true), np.max(lbl_target))
        print(f"max label number: {max_label_number}")

        validate_count = 0.0

        lowest_iou = 9999
        lowest_acc = 9999
        lowest_class = 9999

        iou_sum = 0.0
        acc_sum = 0.0

        accuracy_list = []
        iou_list = []

        for i in range(max_label_number + 1):
            lbl_true_i = lbl_true == i
            lbl_target_i = lbl_target == i
            assert label_names_true[i] == label_names_target[i], "label name must be same"
            intersection_region = np.sum(np.logical_and(lbl_true_i, lbl_target_i))
            union_region = np.sum(np.logical_or(lbl_true_i, lbl_target_i))
            true_region_sum = np.sum(lbl_true_i)
            validate_count += 1
            accuracy = round(intersection_region / true_region_sum, ROUND_NUMBER)
            iou = round(intersection_region / union_region, ROUND_NUMBER)

            accuracy_list.append(accuracy)
            iou_list.append(iou)

            acc_sum += accuracy
            iou_sum += iou

            if iou < lowest_iou:
                lowest_iou = round(iou, ROUND_NUMBER)
                lowest_acc = round(accuracy, ROUND_NUMBER)
                lowest_class = i

            print(
                f"[{i}:{label_names_true[i]}] class accuracy: {accuracy:.3f}% ({intersection_region}/{true_region_sum})")
            print(f"[{i}:{label_names_true[i]}] class iou: {iou:.3f}% ({intersection_region}/{union_region})")

        iou_avg = round(iou_sum * 1.0 / validate_count, ROUND_NUMBER)
        acc_avg = round(acc_sum * 1.0 / validate_count, ROUND_NUMBER)

        iou = lowest_iou
        accuracy = lowest_acc

        result_list.append([len(result_list), osp.basename(true_json), osp.basename(target_json), lowest_class,
                            label_names_true[lowest_class], lowest_acc, lowest_iou, acc_avg, iou_avg])

        total_same_label = lbl_true == lbl_target
        total_accuracy = np.sum(total_same_label) / np.sum(lbl_true == lbl_true)
        # print(f"total_same_label: {total_same_label}")
        # print(f"total_accuracy: {total_accuracy}")

        self.resultDataList.append(ResultData(true_json, target_json, label_names_true, accuracy_list, iou_list))

        # region save image
        good_prefix = f"good({threshold})"
        bad_prefix = f"bad({threshold})"
        if not osp.exists(osp.join(out_dir, good_prefix)):
            os.mkdir(osp.join(out_dir, good_prefix))

        if not osp.exists(osp.join(out_dir, bad_prefix)):
            os.mkdir(osp.join(out_dir, bad_prefix))

        if iou < threshold:
            current_out_dir = osp.join(out_dir, bad_prefix)
        else:
            current_out_dir = osp.join(out_dir, good_prefix)

        utils.lblsave(osp.join(current_out_dir, f"{osp.basename(true_json)}_label_true({iou:.3f}%).png"), lbl_true)
        utils.lblsave(osp.join(current_out_dir, f"{osp.basename(target_json)}_label_target({iou:.3f}%).png"),
                      lbl_target)
        utils.lblsave(osp.join(current_out_dir, f"{osp.basename(target_json)}_label_diff({iou:.3f}%).png"),
                      lbl_true - lbl_target)
        logger.info("Saved to: {}".format(current_out_dir))

    def get_json_file_list(self, true_json_folder_path):
        return glob.glob(os.path.join(true_json_folder_path, "*.json"))
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
   """ app = QApplication(sys.argv)
    MyWindow = LabelValidationWindow()
    MyWindow.show()
    app.exec_()"""

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
