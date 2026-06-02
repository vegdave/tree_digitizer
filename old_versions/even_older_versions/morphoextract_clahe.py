# SO FAR THE BEST PERFORMING VERSION

import cv2
import numpy as np
import networkx as nx
import glob
import os
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from matplotlib import pyplot as plt

# Function to display an image in a window with a wait for a key press
def show_image(title, image, enable=True):
    if not enable:
        return
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(title, 1500, 1000)
    cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image to disk for later review
    save_path = os.path.join('debug_outputs', f"{title.replace(' ', '_')}.jpg")
    os.makedirs('debug_outputs', exist_ok=True)
    cv2.imwrite(save_path, image)
    

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

# Function to find endpoints in a skeleton
def find_endpoints(skeleton):
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    
    skel = (skeleton > 0).astype(np.uint8)
    conv = convolve(skel, kernel, mode='constant', cval=0)
    
    endpoints = (conv == 11).astype(np.uint8)
    return endpoints


# Endpoint bridging function
def bridge_endpoints_via_dilation(skeleton, max_bridge_dist=10):
    skel = (skeleton > 0).astype(np.uint8)

    # Get endpoints 
    endpoints = find_endpoints(skel)

    # Dilate endpoints
    radius = max_bridge_dist // 2
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)
    )
    dilated = cv2.dilate(endpoints, kernel)

    # Skeletonize the dilated endpoints (this creates bridges)
    bridges = skeletonize(dilated > 0).astype(np.uint8)

    # Combine with original skeleton
    combined = np.logical_or(skel, bridges).astype(np.uint8)

    # Normalize to 0/255
    combined = skeletonize(combined).astype(np.uint8) * 255

    return combined

input_folder = 'data/final_tree_samples'
output_folder = 'data/skeletonised_samples'
image_paths = glob.glob(os.path.join(input_folder, '*.jpg'))

if not image_paths:
    raise ValueError("No .jpg files found in the folder")

enable_show_images = True  # Set to False to skip showing images

for path in image_paths:

    img = cv2.imread(path)

    print(f"Processing {path}...")

    if img is None:
        raise ValueError(f"Image failed to load: {path}")

    show_image("Original Image", img, enable_show_images)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(15, 15)).apply(gray)
    show_image("CLAHE Grayscale", gray, enable_show_images)

    # Blur to reduce noise before thresholding
    gray_blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    show_image("Blurred Grayscale", gray_blurred, enable_show_images)

    thresh = cv2.adaptiveThreshold(gray_blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71, 5)

    show_image("Adaptive Threshold", thresh, enable_show_images)

    opening = thresh.copy()  # No morphological opening for now
    """# Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7, 7))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    show_image("Opening", opening, enable_show_images)"""

    """# Connected components to remove small blobs
    num_labels, labels = cv2.connectedComponents(opening, connectivity=8)
    sizes = np.bincount(labels.ravel())

    min_size = 20  # safe starting point

    clean_mask = np.zeros_like(opening)

    for i in range(1, len(sizes)):
        if sizes[i] >= min_size:
            clean_mask[labels == i] = 255

    show_image("Cleaned Mask", clean_mask, enable_show_images)"""

    """kernel = np.ones((2,2), np.uint8)
    reconnected = cv2.dilate(opening, kernel, iterations=1)

    show_image("Reconnected Mask", reconnected, enable_show_images)"""

    # Skeletonize to get 1-pixel wide representation of the tree
    skeleton = skeletonize(opening/255).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)

    overlay = overlay_skeleton_on_gray(gray, cv2.dilate(skeleton*255, kernel, iterations=1))
    show_image("Skeleton Overlay", overlay, enable_show_images)

    skeleton_connected = bridge_endpoints_via_dilation(skeleton, max_bridge_dist=10)
    skeleton_connected = cv2.dilate(skeleton_connected, kernel, iterations=1)

    overlay = overlay_skeleton_on_gray(gray, skeleton_connected)
    show_image("Skeleton Connected Overlay", overlay, enable_show_images)

    """# Save the skeleton overlay
    output_path_skeleton_connected = os.path.join(output_folder, os.path.basename(path).replace('.jpg', '_skeleton_connected.jpg'))
    cv2.imwrite(output_path_skeleton_connected, overlay)

    # Save disconnected skeleton for comparison
    skeleton_disconnected = cv2.dilate(skeleton, kernel, iterations=1)
    skeleton_overlay = overlay_skeleton_on_gray(gray_clahe, skeleton_disconnected * 255)
    output_path_skeleton = os.path.join(output_folder, os.path.basename(path).replace('.jpg', '_skeleton.jpg'))
    cv2.imwrite(output_path_skeleton, skeleton_overlay)"""

    
    resume = input("Press Enter to continue to the next image, or type 'n' to stop: ")
    if resume.lower() == 'n':
        break


"""# Downsample to speed up processing 
scale = 1024 / gray.shape[1]  # width-based scaling
new_size = (1024, int(gray.shape[0]*scale))
resized = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
show_image("Resized Image", resized)"""
