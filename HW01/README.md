# HW01: Scale Invariant Feature Detection and Image Filtering
`difference of gaussian` 、 `joint biliteral filter`


## Part 1: Scale Invariant Feature Detection
Implement the difference of Gaussain (`DoG`) to process the scale invariant feature detection. 
<div align="center">
  <img src="./part1/output/dog.png" width="700">
</div>
 
### Result

 In this part, two octaves are applying with five gaussian images and four difference of gaussian images querying. 

|octaves | gaussian images of each ocatve|
|:--:|:--:|
|`octave1`| ![image](./part1/output/ocative_1.png)|
|`octave2`| ![image](./part1/output/ocative_2.png)|

scale invariant feature detection: 

<div align="center" align-items="flex-start">
  <img src="./part1/data/1.png" width="300">
  <img src="./part1/output/keypoints.png" width="300">
</div>

## Part 2: Image Filtering

