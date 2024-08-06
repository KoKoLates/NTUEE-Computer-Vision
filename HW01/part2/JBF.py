import cv2
import numpy as np
import numpy.typing as npt


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.pad_w = 3 * self.sigma_s
        self.wndw_size = 6 * self.sigma_s + 1

        self.spatial_kernel = np.exp((1 / (2 * self.sigma_s**2)) * 3)

    def joint_bilateral_filter(self, image: cv2.Mat, guidance: cv2.Mat):
        """"""
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_image = cv2.copyMakeBorder(
            image, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)

        ### TODO ###
        output: np.ndarray = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                self.range_kernel = np.exp(
                    -(1 / (2 * self.sigma_r**2))
                    * np.square(
                        padded_guidance[i + self.pad_w, j + self.pad_w]
                        - padded_guidance[
                            i : i + self.wndw_size, j : j + self.wndw_size
                        ]
                    ).sum(axis=-1)
                )
                weights = self.spatial_kernel * self.range_kernel
                output[i, j] = ().sum() / weights.sum()

        return np.clip(output, 0, 255).astype(np.uint8)
