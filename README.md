<h1 align="center">Pose Estimation with Tensorflow</h1>
<p align="center">Pose Estimation using PoseNet model with Tensorflow and Python</p>


## Intro
Code based on [Tensorflow guide](https://www.tensorflow.org/lite/examples/pose_estimation/overview).


## Files
There are two files in the directory:

1. **posenet_mobilenet.tflite** - The actual Tensorflow Lite model;
2. **tf-PoseNet.py** - The code.

## How to use

Inside the tf-PoseNet.py file you will find the following line:

``` python
cap = cv2.VideoCapture('test6.mp4')
```
Just replace **'test6.mp4'** with your video file or **0** to use webcam.

Run the code.
