import pathlib
import os

default_label_file_path = r"C:\labelme\labels.txt"
default_flag_file_path = r"C:\labelme\flags.txt"


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
            f.write("test\n")

    # create default flag
    if not os.path.exists(default_flag_file_path):
        with open(default_flag_file_path, 'w', encoding="utf-8") as f:
            f.write("__ignore__\n")
            f.write("cat\n")
            f.write("dog\n")
