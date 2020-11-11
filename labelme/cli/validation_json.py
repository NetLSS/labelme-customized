import argparse
import base64
import json
import os
import os.path as osp
import numpy as np
import glob
from enum import Enum
import csv


import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils

"""
json 파일에 이미지 포함 저장이 되어있어야함.
json_to_dataset apc2016_obj3.json -o apc2016_obj3_json

https://wikidocs.net/38037
https://stackoverflow.com/questions/2817264/how-to-get-the-parent-dir-location
https://wdprogrammer.tistory.com/58
"""

ROUND_NUMBER = 4

class Result(Enum):
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

def image_data_load(data_true):
    imageData = data_true.get("imageData")
    img = utils.img_b64_to_arr(imageData)
    return img

def lbl_arr_load(data_true, img_true):
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


def load_label_names(data, img):
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


def save_result_csv(save_full_path, add_mode=False):
    open_mode = 'a' if add_mode else 'w'

    try:
        with open(save_full_path, open_mode, encoding='utf-8', newline='') as writer_csv:
            writer = csv.writer(writer_csv, delimiter=",")
            for row in result_list:
                writer.writerow(row)

    except PermissionError:
        print("[PermissionError] file save fail. if file opened please close file and retry.")

def validate_json_file(true_json, target_json, out_full_path=None, threshold=0.5):
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

    img_true = image_data_load(data_true)
    img_target = image_data_load(data_target)

    label_names_true = load_label_names(data_true, img_true)
    label_names_target = load_label_names(data_target, img_target)

    lbl_true = lbl_arr_load(data_true, img_true)
    lbl_target = lbl_arr_load(data_target, img_target)

    lbl_true = np.array(lbl_true)
    lbl_target = np.array(lbl_target)

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
    for i in range(max_label_number + 1):
        lbl_true_i = lbl_true == i
        lbl_target_i = lbl_target == i
        assert label_names_true[i] == label_names_target[i], "label name must be same"
        intersection_region = np.sum(np.logical_and(lbl_true_i, lbl_target_i))
        union_region = np.sum(np.logical_or(lbl_true_i, lbl_target_i))
        true_region_sum = np.sum(lbl_true_i)
        validate_count += 1
        accuracy = intersection_region / true_region_sum
        iou = intersection_region / union_region

        acc_sum += accuracy
        iou_sum += iou

        if iou < lowest_iou:
            lowest_iou = round(iou, ROUND_NUMBER)
            lowest_acc = round(accuracy, ROUND_NUMBER)
            lowest_class = i

        print(f"[{i}:{label_names_true[i]}] class accuracy: {accuracy:.3f}% ({intersection_region}/{true_region_sum})")
        print(f"[{i}:{label_names_true[i]}] class iou: {iou:.3f}% ({intersection_region}/{union_region})")

    iou_avg = round(iou_sum*1.0/validate_count, ROUND_NUMBER)
    acc_avg = round(acc_sum*1.0/validate_count, ROUND_NUMBER)

    iou = lowest_iou
    accuracy = lowest_acc

    result_list.append([len(result_list), osp.basename(true_json), osp.basename(target_json), lowest_class, label_names_true[lowest_class], lowest_acc, lowest_iou, acc_avg, iou_avg])

    total_same_label = lbl_true == lbl_target
    total_accuracy = np.sum(total_same_label)/np.sum(lbl_true == lbl_true)
    #print(f"total_same_label: {total_same_label}")
    #print(f"total_accuracy: {total_accuracy}")

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
    utils.lblsave(osp.join(current_out_dir, f"{osp.basename(target_json)}_label_target({iou:.3f}%).png"), lbl_target)
    utils.lblsave(osp.join(current_out_dir, f"{osp.basename(target_json)}_label_diff({iou:.3f}%).png"), lbl_true-lbl_target)
    logger.info("Saved to: {}".format(current_out_dir))


    # endregion

def get_json_file_list(true_json_folder_path):
    return glob.glob(os.path.join(true_json_folder_path, "*.json"))

def main():
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("json_file_true")  # 정답지
    parser.add_argument("json_file_target")  # 검증이 필요한 파일
    parser.add_argument("-o", "--out", default=None)  # 저장 경로
    args = parser.parse_args()

    json_file_true = args.json_file_true  # 정답 json
    json_file_target = args.json_file_target  # 임의 json

    true_json_folder_path = r"D:\2020\DS\Project\2020-11-02-labelme\labelme-master\labelme\cli\validataion_example\true_label"
    target_json_folder_path = r"D:\2020\DS\Project\2020-11-02-labelme\labelme-master\labelme\cli\validataion_example\user_label"

    true_json_list = glob.glob(os.path.join(true_json_folder_path, "*.json"))
    # target_json_list = glob.glob(os.path.join(target_json_folder_path, "*.json"))

    not_matched_files = []
    for true_json in true_json_list:
        target_json = osp.join(target_json_folder_path, osp.basename(true_json))
        if not osp.exists(target_json):
            result_list.append([len(result_list), osp.basename(true_json), "None", "None", "None", "None", "None", "None", "None"])
            not_matched_files.append(target_json)
        else:
            validate_json_file(true_json, target_json, args.out)

    save_result_csv(osp.join(args.out,"result_total.csv"))

    print(result_list)
    # validate_json_file(json_file_true, json_file_target, args.out)

"""       
    # region 저장 경로 설정/생성
    if args.out is None:
        out_dir = osp.basename(json_file_true).replace(".", "_")
        out_dir = osp.join(osp.dirname(json_file_true), out_dir)
    else:
        out_dir = args.out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    # endregion

    data_true = json.load(open(json_file_true))
    data_target=json.load(open(json_file_target))

    img_true = image_data_load(data_true)
    img_target = image_data_load(data_target)

    label_names_true = load_label_names(data_true, img_true)
    label_names_target = load_label_names(data_target, img_target)

    lbl_true = lbl_arr_load(data_true, img_true)
    lbl_target = lbl_arr_load(data_target, img_target)

    lbl_true = np.array(lbl_true)
    lbl_target = np.array(lbl_target)

    print(f"lbl_true {lbl_true}")
    print(f"lbl_target {lbl_target}")

    print(f"np.max(lbl_true) {np.max(lbl_true)}")
    print(f"np.max(lbl_target) {np.max(lbl_target)}")
    max_label_number = max(np.max(lbl_true), np.max(lbl_target))
    print(f"max label number: {max_label_number}")

    for i in range(max_label_number+1):
        lbl_true_i = lbl_true == i
        lbl_target_i = lbl_target == i
        assert label_names_true[i] == label_names_target[i], "label name must be same"
        intersection_region = np.sum(np.logical_and(lbl_true_i, lbl_target_i))
        true_region_sum = np.sum(lbl_true_i)
        accuracy = intersection_region / true_region_sum
        print(f"[{i}:{label_names_true[i]}] class accuracy: {accuracy}% ({intersection_region}/{true_region_sum})")


    # region save image
    utils.lblsave(osp.join(out_dir, "label_true.png"), lbl_true)
    utils.lblsave(osp.join(out_dir, "label_target.png"), lbl_target)
    logger.info("Saved to: {}".format(out_dir))
    # endregion
"""

if __name__ == "__main__":
    main()
