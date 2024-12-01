# SAM2 Real-Time Segmentation

This project is based on the source code from the repository [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time.git). All checkpoints and environment setup instructions are followed as per the original repository.

## Overview

In this project, we test the `SAM2ImagePredictor` and `SAM2VideoPredictor` respectively.

### Experiments

#### SAM2ImagePredictor

To test the `SAM2ImagePredictor`, please run the script `sam2_img_debug.py`. The output files will be saved in the `display` directory. Note that the `SAM2ImagePredictor` cannot track the prompts given in the first image.

#### SAM2VideoPredictor

To test the `SAM2VideoPredictor`, please run the script `sam2_video_debug.py`. The output files will be saved in the `display_video` directory. The `SAM2VideoPredictor` can effectively track the point annotations given in the first frame.

## Usage

1. Clone the repository and set up the environment as per the instructions in the [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time.git) repository.
2. Run the respective scripts for testing:
   - For `SAM2ImagePredictor`: `python sam2_img_debug.py`
   - For `SAM2VideoPredictor`: `python sam2_video_debug.py`
3. Check the output files in the `display` and `display_video` directories.

## Conclusion

The `SAM2ImagePredictor` is not capable of tracking the prompts given in the first image, whereas the `SAM2VideoPredictor` can effectively track the point annotations given in the first frame.

For more details, please refer to the original [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time.git) repository.