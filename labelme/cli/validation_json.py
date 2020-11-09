import argparse
import base64
import json
import os
import os.path as osp

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

    json_file_true = args.json_file_true
    json_file_target = args.json_file_target

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

    lbl_true = lbl_arr_load(data_true, img_true)
    lbl_target = lbl_arr_load(data_target, img_target)

    print(lbl_true)
    print(lbl_target)

    utils.lblsave(osp.join(out_dir, "label_true.png"), lbl_true)
    utils.lblsave(osp.join(out_dir, "label_target.png"), lbl_target)



    logger.info("Saved to: {}".format(out_dir))


if __name__ == "__main__":
    main()
