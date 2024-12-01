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

# print(os.getcwd())

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
sam2_checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
model_cfg = "./configs/sam2.1/sam2.1_hiera_s.yaml"
# predictor = build_sam2_image_predictor(model_cfg, sam2_checkpoint, device=device)

sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2)


# Initialize global variables
clicked = []
labels = []
rectangles = []
click_points = []  # List to store click positions and colors
mode = 'rectangle'
ix, iy = -1, -1
drawing = False
last_point_time = 0
delay = 0.2


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def image_overlay(image, segmented_image):
    alpha = 0.6
    beta = 0.4
    gamma = 0

    segmented_image = np.array(segmented_image, dtype=np.float32)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    image = np.array(image, dtype=np.float32) / 255.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image


def get_mask(masks, random_color=False, borders=True):
    mask_image = None
    for i, mask in enumerate(masks):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]

        mask = mask.astype(np.float32)

        if i > 0:
            mask_image += mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        else:
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

    return mask_image

def draw_saved_points(img):
    """Function to draw all saved points on the image"""
    for point, label in zip(clicked, labels):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)  # 1: green, 0: red
        cv2.circle(img, (int(point[0]), int(point[1])), 5, color, -1)
   
    # Draw saved rectangles
    for rect in rectangles:
        cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 2)


def draw(event, x, y, flags, param):
    global ix, iy, drawing, rectangles, clicked, labels, mode, last_point_time

    current_time = time.time()

    if mode == 'point':
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([x, y])
            labels.append(1)
            img = frame.copy()
            draw_saved_points(img)  # Redraw all saved points
            cv2.imshow('ROI Selection', img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked.append([x, y])
            labels.append(0)
            img = frame.copy()
            draw_saved_points(img)  # Redraw all saved points
            cv2.imshow('ROI Selection', img)

    elif mode == 'rectangle':
        if event == cv2.EVENT_MBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img = frame.copy()
                draw_saved_points(img)  # Draw saved points first
                cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('ROI Selection', img)
        elif event == cv2.EVENT_MBUTTONUP:
            drawing = False
            rectangles.append([ix, iy, x, y])
            img = frame.copy()
            draw_saved_points(img)  # Redraw all saved points and rectangles
            cv2.imshow('ROI Selection', img)
        elif event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([x, y])
            labels.append(1)
            img = frame.copy()
            draw_saved_points(img)  # Redraw all saved points
            cv2.imshow('ROI Selection', img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked.append([x, y])
            labels.append(0)
            img = frame.copy()
            draw_saved_points(img)  # Redraw all saved points
            cv2.imshow('ROI Selection', img)

def roi_selection_phase():
    global frame, mode, clicked, labels, rectangles
   
    # Create window for ROI selection
    cv2.namedWindow('ROI Selection')
    cv2.setMouseCallback('ROI Selection', draw)
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break
           
        display_frame = frame.copy()
        draw_saved_points(display_frame)  # Draw all saved points
           
        # Display current mode
        cv2.putText(display_frame, f"Mode: {mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
        cv2.imshow('ROI Selection', display_frame)
       
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow('ROI Selection')
            return True
        elif key == ord('p'):
            mode = 'point'
            print("Switched to point mode")
        elif key == ord('r'):
            mode = 'rectangle'
            print("Switched to rectangle mode")
        elif key == ord('c'):
            clicked = []
            labels = []
            rectangles = []
            print("Cleared ROI selection")
   
    return False


def get_binary_mask(masks):
    """Function to convert mask to binary image (white/black)"""
    h, w = masks[0].shape[-2:]
    binary_mask = np.zeros((h, w, 3), dtype=np.uint8)  # Black background
   
    # Combine all masks into one binary mask
    combined_mask = np.zeros((h, w), dtype=bool)
    for mask in masks:
        combined_mask = combined_mask | mask.astype(bool)  # Convert to bool
   
    # Set True parts to white (255)
    binary_mask[combined_mask] = [255, 255, 255]
   
    return binary_mask


def print_roi_info():
    """Function to print ROI information"""
    print("\n=== Final ROI Information ===")

    # Print point coordinates and labels
    if clicked:
        print("\nPoint Coordinates and Labels:")
        for i, (point, label) in enumerate(zip(clicked, labels)):
            label_type = "Positive" if label == 1 else "Negative"
            print(f"Point {i+1}: coordinates={point}, label={label} ({label_type})")
    else:
        print("\nNo points selected")
   
    # Print rectangle coordinates
    if rectangles:
        print("\nRectangle Coordinates:")
        for i, rect in enumerate(rectangles):
            print(f"Rectangle {i+1}: top-left=({rect[0]}, {rect[1]}), bottom-right=({rect[2]}, {rect[3]})")
            print(f"            width={abs(rect[2]-rect[0])}, height={abs(rect[3]-rect[1])}")
    else:
        print("\nNo rectangles selected")
   
    print("\n===================")


def segmentation_phase():
    cv2.namedWindow('Segmentation Result')

    # Variables for calculating FPS and inference time
    frame_count = 0
    total_inference_time = 0
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break
           
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        # SAM2 inference time start
        inference_start = time.time()
        predictor.set_image(frame_rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array(clicked) if clicked else None,
            point_labels=np.array(labels) if labels else None,
            box=rectangles if rectangles else None,
            multimask_output=False,
        )
        inference_time = (time.time() - inference_start) * 1000  # Convert to ms

        # Calculate average
        frame_count += 1
        total_inference_time += inference_time

        # Create and overlay mask
        if masks is not None and len(masks) > 0:
            rgb_mask = get_mask(masks, borders=False)
            frame = image_overlay(frame_rgb, rgb_mask)
            frame = (frame * 255).astype(np.uint8)

            # Create and display binary mask
            binary_mask = get_binary_mask(masks)
            cv2.imshow('Binary Mask', binary_mask)

        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
        cv2.imshow('Segmentation Result', frame)
        out.write(frame)
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            avg_inference = total_inference_time / frame_count
            print("\n=== Performance Statistics ===")
            print(f"Average Inference Time: {avg_inference:.1f}ms")
            print(f"Total Frames Processed: {frame_count}")
            print("===========================\n")
            break


# Main execution code
if __name__ == "__main__":
    # Load SAM2 model
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2)

    # Video capture settings
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        exit()

    # Video save settings
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30.0,
                         (frame_width, frame_height))

    # Execute ROI selection phase
    if roi_selection_phase():
        print_roi_info()
        # If ROI is selected, execute segmentation phase
        segmentation_phase()

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()