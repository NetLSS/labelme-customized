import pathlib
import os

from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

default_label_file_path = r"C:\labelme\labels.txt"
default_flag_file_path = r"C:\labelme\flags.txt"

default_flag_color = Qt.cyan

is_show_flag_default_message = True
is_show_label_default_message = True

def make_dir_tree(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)


def default_init():
    # create default path
    if not os.path.exists(os.path.dirname(default_label_file_path)):
        make_dir_tree(os.path.dirname(default_label_file_path))

    # create default label
    if not os.path.exists(default_label_file_path):
        with open(default_label_file_path, 'w', encoding="utf-8") as f:
            f.write("__ignore__\n")
            f.write("_background_\n")
            f.write("defect\n")

    # create default flag
    if not os.path.exists(default_flag_file_path):
        with open(default_flag_file_path, 'w', encoding="utf-8") as f:
            f.write("__ignore__\n")
            f.write("class1\n")
            f.write("class2\n")
