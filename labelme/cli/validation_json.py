import argparse
import base64
import json
import os
import os.path as osp
import numpy as np

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils

"""
json 파일에 이미지 포함 저장이 되어있어야함.
json_to_dataset apc2016_obj3.json -o apc2016_obj3_json
"""

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


def validate_json_file(true_json, target_json, out_full_path=None):
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

    print(f"lbl_true {lbl_true}")
    print(f"lbl_target {lbl_target}")

    print(f"np.max(lbl_true) {np.max(lbl_true)}")
    print(f"np.max(lbl_target) {np.max(lbl_target)}")
    max_label_number = max(np.max(lbl_true), np.max(lbl_target))
    print(f"max label number: {max_label_number}")

    for i in range(max_label_number + 1):
        lbl_true_i = lbl_true == i
        lbl_target_i = lbl_target == i
        assert label_names_true[i] == label_names_target[i], "label name must be same"
        intersection_region = np.sum(np.logical_and(lbl_true_i, lbl_target_i))
        true_region_sum = np.sum(lbl_true_i)
        accuracy = intersection_region / true_region_sum
        print(f"[{i}:{label_names_true[i]}] class accuracy: {accuracy}% ({intersection_region}/{true_region_sum})")

    # region save image
    utils.lblsave(osp.join(out_dir, f"{osp.basename(true_json)}_label_true.png"), lbl_true)
    utils.lblsave(osp.join(out_dir, f"{osp.basename(target_json)}_label_target.png"), lbl_target)
    logger.info("Saved to: {}".format(out_dir))
    # endregion

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

    validate_json_file(json_file_true, json_file_target, args.out)

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
