import cv2
import numpy as np
import networkx as nx
import glob
import os
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt

# Function to display an image in a window with a wait for a key press
def show_image(title, image):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(title, 1500, 1000)
    cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Show the skeleton overlaid on the original grayscale image
def overlay_skeleton_on_gray(gray_image, skeleton):
    """
    Overlay a skeleton in green on top of a grayscale image.

    Parameters:
        gray_image (np.ndarray): 2D uint8 grayscale image.
        skeleton (np.ndarray): 2D binary image (0/255) of skeleton.

    Returns:
        np.ndarray: BGR image with skeleton overlaid in green.
    """
    # Back to 3-channel BGR
    if len(gray_image.shape) == 2:
        bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = gray_image.copy()

    # Create a green mask where skeleton exists
    green_mask = np.zeros_like(bgr)
    green_mask[skeleton > 0] = [0, 255, 0]  # BGR green

    # Overlay: use max to preserve brightness of original image
    overlay = cv2.addWeighted(bgr, 1.0, green_mask, 1.0, 0)

    return overlay


input_folder = 'data/final_tree_samples'
output_folder = 'data/preprocessed_samples'
image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))

if not image_paths:
    raise ValueError("No .jpg files found in the folder")


path = image_paths[17]  # Process the first image for demonstration

img = cv2.imread(path)

print(f"Processing {path}...")

if img is None:
    raise ValueError(f"Image failed to load: {path}")

show_image("Original Image", img)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(gray, 255,
	cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71, 5)

show_image("Adaptive Threshold", thresh)


# Clean up the binary mask using morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

show_image("Closed Mask", closing)


# Remove small noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1)

show_image("Tree", opening)


# Connected components to remove small blobs
num_labels, labels = cv2.connectedComponents(opening, connectivity=8)
sizes = np.bincount(labels.ravel())

min_size = 20  # safe starting point

clean_mask = np.zeros_like(opening)

for i in range(1, len(sizes)):
    if sizes[i] >= min_size:
        clean_mask[labels == i] = 255

show_image("Cleaned Mask", clean_mask)

kernel = np.ones((2,2), np.uint8)
reconnected = cv2.dilate(clean_mask, kernel, iterations=1)

show_image("Reconnected Mask", reconnected)

# Skeletonize to get 1-pixel wide representation of the tree
skeleton = skeletonize(reconnected // 255).astype(np.uint8) * 255

kernel = np.ones((3,3), np.uint8)
skeleton = cv2.morphologyEx(skeleton, cv2.MORPH_CLOSE, kernel, iterations=1)

# View only the skeletonized image to check for connectivity
skeleton = cv2.dilate(skeleton, kernel, iterations=1)
gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)

overlay = overlay_skeleton_on_gray(gray_clahe, skeleton)
show_image("Skeleton Overlay", overlay)


"""# Downsample to speed up processing 
scale = 1024 / gray.shape[1]  # width-based scaling
new_size = (1024, int(gray.shape[0]*scale))
resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
show_image("Resized Image", resized)"""
