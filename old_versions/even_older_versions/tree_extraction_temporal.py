import cv2
import numpy as np
import glob
import os
from skimage.morphology import skeletonize


# Function to display an image in a window with a wait for a key press
def show_image(title, image, save_debug=False, tip=None):
    
    # Convert grayscale to BGR for visualization if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
    if tip is not None:
        # Draw a red circle at the tip location for visualization
        cv2.circle(image, tip, 6, (0, 0, 255), 2, cv2.LINE_AA)
                
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(title, 1500, 1000)
    cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Get file version for debugging
    folder_name_version = os.path.basename(__file__).split("v")[-1].split("_")[0]
    
    
    # Add just the python file name to the saved image title for easier debugging and review of results
    title = title
       
    # Save the image to disk for later review
    if save_debug:
        print(f"Saving debug image: {title}")
        save_path = os.path.join('debug_outputs/' + folder_name_version, f"{title.replace(' ', '_')}.jpg")
        os.makedirs('debug_outputs/' + folder_name_version, exist_ok=True)
        cv2.imwrite(save_path, image)


def load_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))

    images = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)

    return images, paths


# Find the needle tip
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
    #if show:
        #show_image("Detected Tip", orig_image, tip=tip_global, save_debug=False)

    return tip_global


# Run the full needle tip detection pipeline
def needle_tip_detection_pipeline(gray, s_r=0.5):
    clahe = cv2.createCLAHE(
        clipLimit=7.0,
        tileGridSize=(12,12)
    ) 
    gray_clahe = clahe.apply(gray)
    

    #show_image("Grayscale Image with CLAHE", gray_clahe, save_debug=False)

    # Simple thresholding to get needle mask
    _, binary_mask = cv2.threshold(gray_clahe, 100, 255, cv2.THRESH_BINARY_INV)  # adjust threshold as needed

    #show_image("Binary Mask", binary_mask, save_debug=False)
    
    # Big opening to disconnect branches and isolate the needle
    # Clears up some branches but also erases a bit of the needle tip. There are cases where this is still sufficient - big AC bushes 
    # This can potenitally be improved if we overlay multiple images from the same set of experiments. Good enough for now.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6, 6))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    clean_needle = binary_mask.copy()
    # clean_needle = clean_needle_mask(binary_mask, min_area_frac=0.0005) # currently unused - as unnecessary

    tip = find_needle_tip(clean_needle, gray, search_ratio=s_r)
    
    return tip


# Align frames with needle tip
def align_frames(frames):
    aligned = [frames[0]]
    ref = frames[0]

    for i in range(1, len(frames)):
        shift = cv2.phaseCorrelate(
            np.float32(ref),
            np.float32(frames[i])
        )[0]

        dx, dy = shift
        M = np.float32([[1, 0, -dx],
                        [0, 1, -dy]])

        warped = cv2.warpAffine(frames[i], M,
                                 (frames[i].shape[1], frames[i].shape[0]),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

        aligned.append(warped)

    return aligned


def accumulate_canvas(frames, tips):
    ref_tip = tips[0]

    h, w = frames[0].shape
    canvas = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    for i, (f, tip) in enumerate(zip(frames, tips)):

        # Round to avoid jitter
        dx = int(round(ref_tip[0] - tip[0]))
        dy = int(round(ref_tip[1] - tip[1]))

        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])

        shifted = cv2.warpAffine(
            f.astype(np.float32),
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        mask = (shifted > 0).astype(np.float32)

        canvas += shifted * mask
        weight += mask

    # avoid divide-by-zero
    weight[weight == 0] = 1.0

    return canvas / weight



def temporal_persistence(frames):
    frames = [f.astype(np.float32) for f in frames]

    mean = np.mean(frames, axis=0)
    var = np.var(frames, axis=0)

    # convert variance to stability (continuous, no threshold)
    stability = 1.0 / (1.0 + var)

    # optionally weight by mean intensity (helps suppress flat noise)
    persistence = stability * (mean / 255.0)

    return persistence


def detect_tips(frames):
    tips = []
    for i, f in enumerate(frames):

        # IMPORTANT: your function expects grayscale + mask pipeline inside
        tip = needle_tip_detection_pipeline(f, s_r=0.5)

        tips.append(tip)

        #show_image(f"Tip {i}", f, tip=tip)

    return tips


def pad_frames(frames, pad=300):
    out = []
    for f in frames:
        padded = cv2.copyMakeBorder(
            f,
            pad, pad, pad, pad,
            cv2.BORDER_CONSTANT,
            value=0
        )
        out.append(padded)
    return out


def align_to_reference(frames, tips):
    ref_tip = tips[0]

    aligned = []

    for f, tip in zip(frames, tips):

        dx = ref_tip[0] - tip[0]
        dy = ref_tip[1] - tip[1]

        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])

        warped = cv2.warpAffine(
            f,
            M,
            (f.shape[1], f.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        aligned.append(warped)

    return aligned


def show_aligned_sequence(frames, title="Aligned Frame", delay=200):
    for i, f in enumerate(frames):
        vis = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imshow(f"{title} {i}", vis)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()


def persistence_map(frames, sigma=15.0):
    frames = [f.astype(np.float32) for f in frames]

    acc = np.zeros_like(frames[0], dtype=np.float32)

    for t in range(1, len(frames)):
        diff = np.abs(frames[t] - frames[t - 1])
        stable = np.exp(-diff / sigma)

        acc += stable

    acc /= (len(frames) - 1)
    return acc


def show_map(persistence):
    norm = cv2.normalize(persistence, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    show_image("Persistence Map", norm)


main_path = "data/full_sets/S3"
frames, paths = load_images(main_path)

# tip detection (your method)
tips = detect_tips(frames)

persistence = accumulate_canvas(frames, tips)

norm = cv2.normalize(persistence, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

show_image("Persistance map", norm, save_debug=True)

# Apply clahe to norm
gray = cv2.createCLAHE(clipLimit=7.0, tileGridSize=(13, 13)).apply(norm)
show_image("CLAHE Grayscale", gray)
