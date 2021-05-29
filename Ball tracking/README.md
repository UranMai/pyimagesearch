# Ball tracking

This script allows to track ball movement. The process for tracking is as follows:
- Detect the green ball using lower and upper color boundaries
- Traverse video frames create pixel masks, erode and dilate
- Find contours and draw circle center
- Draw line between prev and current center points
- Create GIF animation of first frames

### Run
```sh
python ball_tracking.py --video ball_tracking_example.mp4 --buffer 30
```

### Sources
1. [Ball tracking OpenCV](https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/)
2. [GIF creation](https://pysource.com/2021/03/25/create-an-animated-gif-in-real-time-with-opencv-and-python/)

### GIF
<img src="https://github.com/UranMai/pyimagesearch/blob/main/Ball%20tracking/ball.gif" />
