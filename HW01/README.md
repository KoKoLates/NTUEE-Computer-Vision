# HW01: Scale Invariant Feature Detection and Image Filtering
`difference of gaussian` 、 `biliteral filter`


## Part 1: Scale Invariant Feature Detection
Implement difference of Gaussain (DoG).
<div align="center">
  <img src="./part1/output/dog.png" width="600">
</div>
 

|octaves | Gaussian Images each ocatves|
|:--:|:--:|
|`octave1`| ![image](./part1/output/ocative_1.png)|
|`octave2`| ![image](./part1/output/ocative_2.png)|

```
usage: main.py [-h] [--threshold THRESHOLD]
               [--image_path IMAGE_PATH] [--save SAVE]

main function of Difference of Gaussian

options:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        threshold value for feature selection
  --image_path IMAGE_PATH
                        path to input image
  --save SAVE           save the image with keypoints plotting
```

```bash
python main.py
```

<div align="center" align-items="flex-start">
  <img src="./part1/data/1.png" width="300">
  <img src="./part1/output/keypoints.png" width="300">
</div>

## Part 2: Image Filtering

