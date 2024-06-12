import cv2
import argparse
import numpy as np

from DoG import Difference_of_Gaussian
from DoG import np_arr_f32, np_arr_i32


def plot_keypoints(img_gray, keypoints, save_path: str) -> None:
    img = np.repeat(np.expand_dims(img_gray, axis=2), 3, axis=2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)


def main() -> None:
    print(f"Processing {args.image_path}")
    image: np_arr_f32 = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    kps: np_arr_i32 = Difference_of_Gaussian(args.threshold).get_keypoints(image)
    if (args.save):
        plot_keypoints(image, kps, "./output/keypoints.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="main function of Difference of Gaussian"
    )
    parser.add_argument(
        "--threshold",
        default=5.0,
        type=float,
        help="threshold value for feature selection",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/1.png",
        help="path to input image",
    )
    parser.add_argument(
        "--save",
        default=True,
        type=bool,
        help="save the image with keypoints plotting"
    )
    args = parser.parse_args()
    main()
