import cv2
import numpy as np
import numpy.typing as npt

np_arr_f32 = npt.NDArray[np.float32]
np_arr_i32 = npt.NDArray[np.int32]
np_arr_ui8 = npt.NDArray[np.uint8]


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r: np_arr_f32 = sigma_r
        self.sigma_s: np_arr_f32 = sigma_s
        self.pad_w: np_arr_f32 = 3 * sigma_s
        self.wndw_size: np_arr_f32 = 6 * sigma_s + 1

    def joint_bilateral_filter(self, img: cv2.Mat, guidance: cv2.Mat) -> np_arr_ui8:
        padded_img: np_arr_i32 = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, cv2.BORDER_REFLECT
        ).astype(np.int32)
        padded_guidance: np_arr_i32 = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, cv2.BORDER_REFLECT
        ).astype(np.int32)

        ### TODO ###

        return np.clip(output, 0, 255).astype(np.uint8)
