
# Speech2Vid

Implementation of [You said that?](https://www.robots.ox.ac.uk/~vgg/publications/2017/Chung17b/chung17b.pdf) in Python.

### Prerequisites
 - Python3
 - Cuda
 - ffmpeg
 - Scipy
 - OpenCV
 - dlib

### Usage

 1. **Feature extraction:** Extract and store the audio and identity features 
`python3 extract_features.py -d=data/`
 
 2. **Train:** Train the model using the processed data
  `python3 train.py -f=features/`
  
 3. **Generate video:** Generate the output video for a given audio and image input. 
 `python3 predict.py -m=saved_models/best_model.h5 -t=1.mp4 -f=frame.jpg`