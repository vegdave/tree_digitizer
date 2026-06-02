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


# Currently unused - as unnecessary
def clean_needle_mask(binary_mask, min_area_frac=0.0005):
    """
    Keep only the right half of the binary mask, remove small disconnected objects.
    
    Args:
        binary_mask: 2D binary mask (needle + branches)
        min_area_frac: minimum area fraction relative to image size
    
    Returns:
        cleaned_mask: binary mask with only large objects in the right half
    """
    h, w = binary_mask.shape
    min_area = int(h * w * min_area_frac)
    
    # Zero out left half
    binary_mask[:, :w//2] = 0
    
    # Find connected components in right half
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(binary_mask)
    
    for i in range(1, num_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask


# Find the needle tip by skeletonizing the cleaned mask
def find_needle_tip(needle_mask, orig_image, show=True, search_ratio=1.0, open_iter=2):
    """
    Args:
        needle_mask : np.ndarray, uint8
            Needle outline after preprocessing (needle = 255, background = 0)
        orig_image : np.ndarray
            Original grayscale image for visualization
        show : bool
            If True, draws a red circle at the tip and shows using show_image()
        search_ratio : float
            Fraction of width (from the RIGHT side) to process, in (0, 1]. Use 1.0 to process full image.
        open_iter : int
            Morphological opening iterations to reduce speckle noise.

    Returns:
    (int, int)
        Tip (x, y) in full-image coordinates.
    """

    if needle_mask.dtype != np.uint8:
        raise ValueError("needle_mask must be uint8 with values 0/255.")
    if needle_mask.ndim != 2:
        raise ValueError("needle_mask must be a single-channel image.")

    # Limit to right portion of the image based on search_ratio
    h, w = needle_mask.shape
    search_ratio = float(np.clip(search_ratio, 1e-3, 1.0))
    x0 = int(w * (1 - search_ratio))
    roi = needle_mask[:, x0:w]

    # Clean small speckles using morphological opening - dilation followed by erosion
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    
    #show_image("Cleaned after opening", clean)

    # Ensure a single main component (remove small components)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((clean > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        raise ValueError("No foreground found after cleanup.")
    # Keep the largest component by area (ignoring background label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = 1 + int(np.argmax(areas))
    main = (labels == keep_label).astype(np.uint8) * 255
    
    show_image("Largest Component", main)

    # Skeleton
    skel = skeletonize(main > 0)
    if not np.any(skel):
        raise ValueError("Skeleton is empty; try reducing open_iter or widening search_ratio.")

    # Find skeleton endpoints (pixels with exactly 1 8-neighbour)
    skel_u8 = skel.astype(np.uint8)
    # 8-neighbourhood count via convolution
    kernel8 = np.array([[1,1,1],
                        [1,0,1],
                        [1,1,1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, -1, kernel8, borderType=cv2.BORDER_CONSTANT)
    endpoints_y, endpoints_x = np.where((skel_u8 == 1) & (neighbor_count == 1))
    if len(endpoints_x) < 1:
        raise ValueError("No endpoints found on skeleton.")

    # --- PCA on foreground to get major axis direction ---
    ys_fg, xs_fg = np.where(main > 0)
    if len(xs_fg) < 2:
        raise ValueError("Insufficient foreground pixels for PCA.")
    pts = np.column_stack((xs_fg, ys_fg)).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(pts, mean=None, maxComponents=2)
    v = eigenvectors[0]  # principal direction (unit-ish vector)
    v = v / np.linalg.norm(v)  # normalize to unit length

    # For consistency, point the axis roughly to the RIGHT (positive x).
    # If v points left, flip it so positive projection is "to the right".
    if v[0] < 0:
        v = -v

    # Project endpoints on the major axis (in ROI coords)
    # We want the endpoint with the MINIMUM projection -> "leftmost along the axis".
    # Use centered coordinates (subtract mean) to be robust.
    mean_xy = mean.ravel()  # (mx, my) in ROI coords
    proj_vals = []
    endpoint_coords = []
    for ex, ey in zip(endpoints_x, endpoints_y):
        p = np.array([ex, ey], dtype=np.float32)
        proj = np.dot((p - mean_xy), v)  # scalar projection onto major axis
        proj_vals.append(proj)
        endpoint_coords.append((int(ex), int(ey)))

    # Leftmost along major axis = smallest projection
    idx = int(np.argmin(proj_vals))
    tip_local = endpoint_coords[idx]  # (x, y) in ROI coords

    # Convert to full-image coordinates
    tip_global = (tip_local[0] + x0, tip_local[1])

    # Visualise (single red circle) on original grayscale image
    if show:
        overlay = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2BGR)
        cv2.circle(overlay, tip_global, 6, (0, 0, 255), 2, cv2.LINE_AA)
        show_image("Detected Tip", overlay)

    return tip_global


folder = 'data/final_tree_samples'
image_paths = glob.glob(os.path.join(folder, '*.jpg'))

if not image_paths:
    raise ValueError("No .jpg files found in the folder")

for path in image_paths:
    img = cv2.imread(path)
    
    print(f"Processing {path}...")
    
    if img is None:
        raise ValueError(f"Image failed to load: {path}")

    #show_image("Original Image", img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=7.0,
        tileGridSize=(12,12)
    )
    gray = clahe.apply(gray)

    show_image("Grayscale Image with CLAHE", gray)

    # Simple thresholding to get needle mask
    _, binary_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # adjust threshold as needed
    
    show_image("Binary Mask", binary_mask)
    
    # Big opening to disconnect branches and isolate the needle
    # Clears up some branches but also erases a bit of the needle tip. There are cases where this is still sufficient - big AC bushes 
    # This can potenitally be improved if we overlay multiple images from the same set of experiments. Good enough for now.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6, 6))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    clean_needle = binary_mask.copy()
    # clean_needle = clean_needle_mask(binary_mask, min_area_frac=0.0005) # currently unused - as unnecessary

    tip = find_needle_tip(clean_needle, gray, search_ratio=0.5)

    
