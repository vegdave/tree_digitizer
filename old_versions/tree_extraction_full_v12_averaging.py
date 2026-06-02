import cv2
import numpy as np
import networkx as nx
import glob
import os
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from matplotlib import pyplot as plt

# Function to display an image in a window with a wait for a key press
def show_image(title, image, save_debug=False, tip=None):

    if not ENABLE_SHOW_IMAGES:
        return
    
    # Convert grayscale to BGR for visualization if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    if tip is not None:
        # Draw a red circle at the tip location for visualization
        cv2.circle(image, tip, 6, (0, 0, 255), 2, cv2.LINE_AA)
                
    #cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    #cv2.resizeWindow(title, 1500, 1000)
    #cv2.imshow(title, image)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # Get file version for debugging
    folder_name_version = os.path.basename(__file__).split("v")[-1].split("_")[0]
    
    
    # Add current just the python file name to the saved image title for easier debugging and review of results
    title = CURRENT_IMAGE + title
       
    # Save the image to disk for later review
    if save_debug:
        print(f"Saving debug image: {title}")
        save_path = os.path.join('debug_outputs/' + folder_name_version, f"{title.replace(' ', '_')}.jpg")
        os.makedirs('debug_outputs/' + folder_name_version, exist_ok=True)
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
    
    endpoints = (conv == 11).astype(np.uint8)   # 10 from the center pixel + 1 from exactly one neighbour = 11
    return endpoints


# Endpoint bridging function
# TODO REMOVE IF smart connect works better
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
    
    #show_image("Largest Component", main)

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
        show_image("Detected Tip", orig_image, tip=tip_global)

    return tip_global


# Run the full needle tip detection pipeline
def needle_tip_detection_pipeline(gray, s_r=0.5):
    clahe = cv2.createCLAHE(
        clipLimit=7.0,
        tileGridSize=(12,12)
    ) 
    gray_clahe = clahe.apply(gray)

    show_image("Grayscale Image with CLAHE", gray_clahe)

    # Simple thresholding to get needle mask
    _, binary_mask = cv2.threshold(gray_clahe, 100, 255, cv2.THRESH_BINARY_INV)  # adjust threshold as needed

    show_image("Binary Mask", binary_mask)
    
    # Big opening to disconnect branches and isolate the needle
    # Clears up some branches but also erases a bit of the needle tip. There are cases where this is still sufficient - big AC bushes 
    # This can potenitally be improved if we overlay multiple images from the same set of experiments. Good enough for now.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6, 6))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    clean_needle = binary_mask.copy()
    # clean_needle = clean_needle_mask(binary_mask, min_area_frac=0.0005) # currently unused - as unnecessary

    tip = find_needle_tip(clean_needle, gray, search_ratio=s_r)
    
    return tip
    

# Adjust the tip of the needle to the skeleton
def snap_to_skeleton(skeleton, tip, radius=20):
    skel_patch = skeleton[tip[1]-radius:tip[1]+radius+1, tip[0]-radius:tip[0]+radius+1]
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
    skel_patch.astype(np.uint8), connectivity=8
    )       

    areas = stats[1:, cv2.CC_STAT_AREA]
    
    if areas.size == 0:
        print("No skeleton components found in patch; returning original tip.")
        return tip
    
    largest = 1 + int(np.argmax(areas))
    mask = (labels == largest)
    
    ys, xs = np.where(mask)

    # convert to global coords
    xs_global = xs + (tip[0] - radius)
    ys_global = ys + (tip[1] - radius)

    # Find the skeleton pixel closest to the original tip
    dists_sqr = (xs_global - tip[0])**2 + (ys_global - tip[1])**2
    idx = np.argmin(dists_sqr)

    start = (xs_global[idx], ys_global[idx])
    
    return start


# Extract the skeleton pixels connected to the tip
def connected_skeleton_component(skeleton, tip):
    # Make a copy to avoid modifying the original
    skel_img = (skeleton * 255).astype(np.uint8)

    # OpenCV requires a mask 2 pixels larger in each dimension
    mask = np.zeros((skel_img.shape[0] + 2, skel_img.shape[1] + 2), np.uint8)

    # Flood-fill starting at the tip
    seed = (tip[0], tip[1])
    fill_value = 128  # must not conflict with 0 or 255
    cv2.floodFill(skel_img, mask, seed, fill_value, flags=8)

    # Extract the filled component as a binary mask
    connected_component = (skel_img == fill_value).astype(np.uint8) * 255

    return connected_component


# Run the full tree extraction pipeline after finding the needle tip
def branch_extraction_pipeline(gray, tip):
    # Apply CLAHE to enhance contrast
    gray = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(15, 15)).apply(gray)
    show_image("CLAHE Grayscale", gray)

    # Blur to reduce noise before thresholding
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    show_image("Blurred Grayscale", gray)
    
    thresh = cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, 5)       # 71, 5 initially but needs fine tuning. Maybe denoising allows more aggressive parameters?
    show_image("Adaptive Threshold", thresh)

    ###### Opening/Closing to remove noise #######
    #TODO fine tune this        
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)), iterations=2)
    #show_image("Closed Threshold", thresh)

    # Skeletonize to get 1-pixel wide representation of the tree
    skeleton = skeletonize(thresh/255).astype(np.uint8)

    kernel = np.ones((3,3), np.uint8)
    overlay = overlay_skeleton_on_gray(gray, cv2.dilate(skeleton*255, kernel, iterations=1))
    show_image("Skeleton Overlay", overlay)

    # Bridging also needs finetuning - too much bridging can cause false connections or noise amp
    #TODO also fine tune this
    #skeleton_connected = bridge_endpoints_via_dilation(skeleton, max_bridge_dist=10)

    #overlay = overlay_skeleton_on_gray(gray, cv2.dilate(skeleton_connected, kernel, iterations=1))
    #show_image("Bridged Skeleton Overlay", overlay)
    
    adjusted_tip = snap_to_skeleton(skeleton, tip, radius=20)
    
    show_image("Adjusted Tip on Skeleton", overlay, tip=adjusted_tip)
    
    # Keep only the skeleton pixels connected to the tip
    main_tree = connected_skeleton_component(skeleton, adjusted_tip)
    overlay = overlay_skeleton_on_gray(gray, cv2.dilate(main_tree, kernel, iterations=1))
      
    show_image("Extracted Branches Overlay", overlay, tip=adjusted_tip)
     
    """skeleton_connected = bridge_endpoints_via_dilation(skeleton, max_bridge_dist=10)
    skeleton_connected = cv2.dilate(skeleton_connected, kernel, iterations=1)

    overlay = overlay_skeleton_on_gray(gray, skeleton_connected)
    show_image("Skeleton Connected Overlay", overlay)"""
    
    return skeleton


# Average a list of images using OpenCV's accumulate function to reduce noise and enhance common features across multiple samples
def average_images_cv(image_list):
    acc = np.zeros_like(image_list[0], dtype=np.float32)

    for img in image_list:
        cv2.accumulate(img, acc)

    avg = acc / len(image_list)
    return cv2.convertScaleAbs(avg)


# Median stacking if averaging is producing ghosts (non stationary camera). More memory and compute intensive but can be more robust to outliers and misalignments.
def median_images(image_list):
    stack = np.stack(image_list, axis=0)
    median = np.median(stack, axis=0)
    return median.astype(np.uint8)


ENABLE_SHOW_IMAGES = True  # Set to False to skip showing images
DENOISING = True  # Set to False to skip averaging/median stacking and use raw images (more noise but no ghosting)

folder = 'data/full_sets/S3_final_tree_extract'
#folder = 'data/final_tree_samples'
image_paths = glob.glob(os.path.join(folder, '*.jpg'))

if not image_paths:
    raise ValueError("No .jpg files found in the folder")

if DENOISING:
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray)
        
    # Image name without extension for display and saving
    CURRENT_IMAGE = os.path.splitext(os.path.basename(path))[0]

    show_image("Sample Grayscale Image", images[-1])

    # Optional: Average the images to enhance common features and reduce noise - floating dust. Can be either average or median, test other methods if ghosts or blurring is an issue
    combined = average_images_cv(images)
    show_image("Combined Average Image", combined)

    tip = needle_tip_detection_pipeline(combined, s_r=0.5)

    branch_extraction_pipeline(combined, tip)
else:
    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Image name without extension for display and saving
        CURRENT_IMAGE = os.path.splitext(os.path.basename(path))[0]

        show_image("Sample Grayscale Image", gray)

        tip = needle_tip_detection_pipeline(gray, s_r=0.5)

        branch_extraction_pipeline(gray, tip)