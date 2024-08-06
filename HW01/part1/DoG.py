import cv2
import numpy as np

import numpy.typing as npt

np_arr_f32 = npt.NDArray[np.float32]
np_arr_i32 = npt.NDArray[np.int32]


class Difference_of_Gaussian(object):
    def __init__(self, threshold: float):
        self.threshold: float = threshold
        self.sigma: float = 2 ** (1 / 4)
        self.num_octaves: int = 2
        self.num_DoG_images_per_octave: int = 4
        self.num_guassian_images_per_octave: int = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image: np_arr_f32) -> np_arr_i32:
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur(kernel = (0, 0), sigma = self.sigma**___)
        self.gaussian_images: list[np_arr_f32] = []
        for _ in range(self.num_octaves):
            self.gaussian_images.append(image)
            for i in range(1, self.num_guassian_images_per_octave):
                self.gaussian_images.append(
                    cv2.GaussianBlur(image, (0, 0), self.sigma**i)
                )

            # down sampling to half for next octave
            h, w = image.shape
            image = cv2.resize(
                self.gaussian_images[-1],
                (w // 2, h // 2),
                interpolation=cv2.INTER_NEAREST,
            )
        
        self._plot_gaussian_images()

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images: list[np_arr_f32] = []
        for i in range(self.num_octaves):
            n: int = i * self.num_guassian_images_per_octave
            for j in range(self.num_DoG_images_per_octave):
                dog_images.append(
                    cv2.subtract(
                        self.gaussian_images[n + j + 1], self.gaussian_images[n + j]
                    )
                )

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #   Keep local extremum as a keypoint
        keypoints: list[np_arr_i32] = []
        for i in range(self.num_octaves):
            n: int = self.num_DoG_images_per_octave
            images: np_arr_f32 = np.stack(dog_images[n * i : n * (i + 1)], axis=0)
            keypoints.append(self._find_local_extremum(images) * 2**i)

        keypoints = np.vstack(keypoints)

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints

    def _find_local_extremum(self, images: np_arr_f32) -> np_arr_i32:
        """Check the pixel value is local extremum (local maxima or local minima),
        with a stack of images. compare with the cubic pixel (26 pixels) of the
        difference of Gaussian images stack
        @param images: a stack of difference of Gaussian images stack
        @return: the `(x,y)` coordinates of local extremum
        """
        keypoints: list[list] = []
        b, h, w = images.shape
        for i in range(b - 2):
            for x in range(w - 2):
                for y in range(h - 2):
                    center = images[i + 1, y + 1, x + 1]
                    if abs(center) < self.threshold:
                        continue

                    local_min = images[i : i + 3, y : y + 3, x : x + 3].min()
                    local_max = images[i : i + 3, y : y + 3, x : x + 3].max()
                    if (center == local_max) or (center == local_min):
                        keypoints.append([y + 1, x + 1])

        keypoints = np.array(keypoints, dtype=np.int32).reshape(-1, 2)
        return keypoints

    def _plot_gaussian_images(self) -> None:
        images: list[np_arr_f32] = [image.copy() for image in self.gaussian_images]
        n: int = self.num_guassian_images_per_octave

        for i in range(self.num_octaves):
            concat = cv2.hconcat(images[i * n: (i + 1) * n])
            cv2.imwrite(f"./output/ocative_{i + 1}.png", concat)
