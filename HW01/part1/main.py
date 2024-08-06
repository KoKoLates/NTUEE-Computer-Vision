import cv2
import argparse
import numpy as np

from DoG import Difference_of_Gaussian
from DoG import np_arr_f32, np_arr_i32


def plot_keypoints(img_gray: np_arr_f32, keypoints: np_arr_i32, save_path: str) -> None:
    """plot the keypoints on the gray-scale input image
    @param img_gray: the input image with gray-scale
    @param keypoints: keypoints finding by difference of guassian
    @param save_path: the file path of image for saving
    """
    img = np.repeat(np.expand_dims(img_gray, axis=2), 3, axis=2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)


def main() -> None:
    print(f"Processing {args.image_path}")
    image: np_arr_f32 = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    # find out the keypoints with difference of gaussian
    kps: np_arr_i32 = Difference_of_Gaussian(args.threshold).get_keypoints(image)

    if args.save:
        plot_keypoints(image, kps, "./output/keypoints.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main function of Difference of Gaussian")
    parser.add_argument("--threshold", default=5.0, type=float, help="threshold value for feature selection")
    parser.add_argument("--image_path", default="./data/1.png", type=str, help="path to input image")
    parser.add_argument("--save", default=True, type=bool, help="save the image with keypoints plotting")
    args = parser.parse_args()
    main()
