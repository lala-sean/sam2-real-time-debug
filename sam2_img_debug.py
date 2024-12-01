import os
import numpy as np
import torch
import cv2
import time
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

import os

import random


class_mapping = {
    "Background": 0,
    "Shaft": 10,
    "Wrist": 20,
    "Claspers": 30,
    "Probe": 40
}


def sample_points(mask, label, num_points=2):
    points = np.argwhere(mask == label)
    if len(points) == 0:
        return []
    sampled_points = points[np.random.choice(points.shape[0], num_points, replace=False)]
    return sampled_points

# Select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# SAM2 model
sam2_checkpoint = "//mnt/iMVR/shuojue/code/segment-anything-2-real-time/checkpoints/sam2_hiera_large.pt"
model_cfg = "//mnt/iMVR/shuojue/code/segment-anything-2-real-time/sam2_configs/sam2_hiera_l.yaml"
# predictor = build_sam2_image_predictor(model_cfg, sam2_checkpoint, device=device)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2)


  

# Read frames from files
frame_folder = '/mnt/iMVR/shuojue/data/instrument_dataset_LND/color'
frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])

if not frame_files:
    print("Error: No frames found in the specified folder")
    exit()

# Get video frame width and height
first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
frame_height, frame_width = first_frame.shape[:2]

# Video save settings
out = cv2.VideoWriter('output.mp4',
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        30.0,
                        (frame_width, frame_height))
# Testing the image predictor
# It indicates that imagepredictor cannot track the prompt; you can only use it when per-frame annotations are given
# Loop through frames

for i, frame_file in enumerate(frame_files):
    frame = cv2.imread(os.path.join(frame_folder, frame_file))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_mask = cv2.imread(os.path.join(frame_folder.replace('color', 'l_mask'), frame_file), cv2.IMREAD_GRAYSCALE)
    if i in [0, 5]:
        clicked = []
        labels = []
        shaft_points = sample_points(frame_mask, class_mapping["Shaft"], 1)
        wrist_points = sample_points(frame_mask, class_mapping["Wrist"], 1)
        for point in shaft_points:
            clicked.append([point[1], point[0]])  # (X, Y) 
            labels.append(1)
        for point in wrist_points:
            clicked.append([point[1], point[0]])  # (X, Y) 
            labels.append(1)
        
        background_points = sample_points(frame_mask, class_mapping["Background"], 2)
        for point in background_points:
            clicked.append([point[1], point[0]])  # (X, Y) 
            labels.append(0)
        rectangles = None
        # visualize the clicked points (foreground in red, background in blue)
        frame_copy = frame_rgb.copy()
        for point, label in zip(clicked, labels):
            color = (255, 0, 0) if label == 1 else (0, 0, 255)
            cv2.circle(frame_copy, tuple(point), 5, color, -1)
            # write the image
            cv2.imwrite(f'display/initial_{i}.png', cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR))
    else:
        rectangles = None
        clicked = None
        labels = None

    # Run SAM2
    inference_start = time.time()
    predictor.set_image(frame_rgb)
    output, scores, _ = predictor.predict(
        point_coords=np.array(clicked) if clicked else None,
        point_labels=np.array(labels) if labels else None,
        box=rectangles if rectangles else None,
        multimask_output=False,
    )
    inference_time = (time.time() - inference_start) * 1000

    # Convert output to numpy array
    
    output = np.transpose(output*255, (1, 2, 0)).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output[...,:2] = 0

    # add output to original frame
    output = cv2.addWeighted(frame, 0.5, output, 0.5, 0)
    # Save output to video
    # out.write(output)
    # Save output to file
    cv2.imwrite(f'display/output_{i}.png', output)

    # Display output
    # cv2.imshow('Output', output)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# # Cleanup
# out.release()
# cv2.destroyAllWindows()