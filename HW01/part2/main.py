import os
import cv2
import argparse
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass

from JBF import Joint_bilateral_filter
from JBF import np_arr_f32


@dataclass
class Settings(object):
    rgb: np_arr_f32
    sigma_s: np_arr_f32
    sigma_r: np_arr_f32


def load_setting(file_path: str) -> Settings:
    with open(file_path, "r") as file:
        s = Settings()

    return s


def main() -> None:
    img = cv2.imread(args.image_path)
    img_rgb: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray: cv2.Mat = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### TODO ###
    s: Settings = load_setting(args.setting_path)

    JBF = Joint_bilateral_filter(s.sigma_s, s.sigma_r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="main function of joint bilateral filter"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/1.png",
        help="path to input image",
    )
    parser.add_argument(
        "--setting_path",
        type=str,
        default="./data/1_setting.txt",
        help="path to setting file",
    )
    args = parser.parse_args()
    main()
