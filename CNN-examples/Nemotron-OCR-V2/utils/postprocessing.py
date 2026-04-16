import numpy as np
import cv2


def rrect_to_quad(rrect, cell_size, x, y):
    """
    Convert FCOS-style rbox to quad (4 corner points)

    rrect format: [dTop, dRight, dBottom, dLeft, theta]
    - dTop/Right/Bottom/Left: distances to edges from grid center
    - theta: rotation angle in radians
    """
    # Grid center point
    cell_off = cell_size / 2
    prior_x = x * cell_size + cell_off
    prior_y = y * cell_size + cell_off

    # Extract rbox parameters
    d_top = rrect[0]
    d_right = rrect[1]
    d_bottom = rrect[2]
    d_left = rrect[3]
    theta = rrect[4]

    # Rotation vectors
    vx_x = np.cos(theta)
    vx_y = np.sin(theta)
    vy_x = np.cos(theta - np.pi/2)
    vy_y = np.sin(theta - np.pi/2)

    # Calculate 4 corners
    # Top-left
    p0_x = prior_x - vx_x * d_left + vy_x * d_top
    p0_y = prior_y - vx_y * d_left + vy_y * d_top

    # Top-right
    p1_x = prior_x + vx_x * d_right + vy_x * d_top
    p1_y = prior_y + vx_y * d_right + vy_y * d_top

    # Bottom-right
    p2_x = prior_x + vx_x * d_right - vy_x * d_bottom
    p2_y = prior_y + vx_y * d_right - vy_y * d_bottom

    # Bottom-left
    p3_x = prior_x - vx_x * d_left - vy_x * d_bottom
    p3_y = prior_y - vx_y * d_left - vy_y * d_bottom

    return [[p0_x, p0_y], [p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y]]


def extract_detections(confidence, rboxes, threshold=0.5):
    """Extract detections using the correct FCOS format"""
    # Remove batch dimension
    conf_map = confidence[0]  # [256, 256]
    rbox_map = rboxes[0]      # [256, 256, 5] or [256, 5, 256] for NHWC models

    # Handle both NCHW (256, 256, 5) and NHWC (256, 5, 256) output formats
    if rbox_map.shape[-1] != 5:
        # NHWC format: (256, 5, 256) -> transpose to (256, 256, 5)
        rbox_map = np.transpose(rbox_map, (0, 2, 1))
        print(f"Transposed rbox_map from NHWC to NCHW format: {rbox_map.shape}")

    # Apply sigmoid to logits
    conf_map = 1 / (1 + np.exp(-conf_map))

    high_conf_mask = conf_map > threshold
    y_indices, x_indices = np.where(high_conf_mask)

    detections = []
    grid_size = 256
    cell_size = 1024 / grid_size

    for y, x in zip(y_indices, x_indices):
        conf = float(conf_map[y, x])
        rrect = rbox_map[y, x]  # [5]: dTop, dRight, dBottom, dLeft, theta

        # Convert to quad using FCOS format
        quad = rrect_to_quad(rrect, cell_size, x, y)

        detections.append({
            'quad': quad,
            'confidence': conf
        })

    return detections


def polygon_iou(poly1, poly2):
    """Calculate IoU between two quadrilaterals using Shapely"""
    from shapely.geometry import Polygon

    p1 = Polygon(poly1)
    p2 = Polygon(poly2)

    if not p1.is_valid or not p2.is_valid:
        return 0.0

    intersection = p1.intersection(p2).area
    union = p1.union(p2).area

    if union == 0:
        return 0.0

    return intersection / union


def non_maximum_suppression(detections, iou_threshold=0.5):
    """
    Apply NMS to remove duplicate/overlapping detections.

    Args:
        detections: List of detection dicts with 'quad' and 'confidence'
        iou_threshold: IoU threshold for suppression (default 0.5 matches PyTorch impl)

    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []

    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while len(detections) > 0:
        # Keep the detection with highest confidence
        current = detections.pop(0)
        keep.append(current)

        # Remove detections that overlap significantly with current
        filtered = []
        for det in detections:
            iou = polygon_iou(current['quad'], det['quad'])
            if iou < iou_threshold:
                filtered.append(det)

        detections = filtered

    return keep


def scale_and_format(detections, original_size):
    """Scale to original image and add normalized coords"""
    orig_h, orig_w = original_size
    scale_x = orig_w / 1024
    scale_y = orig_h / 1024

    results = []
    for det in detections:
        quad = det['quad']
        scaled_quad = [[x * scale_x, y * scale_y] for x, y in quad]

        xs = [p[0] for p in scaled_quad]
        ys = [p[1] for p in scaled_quad]

        results.append({
            'quad': scaled_quad,
            'confidence': det['confidence'],
            'left': min(xs) / orig_w,
            'right': max(xs) / orig_w,
            'upper': min(ys) / orig_h,
            'lower': max(ys) / orig_h
        })

    return results


def draw_boxes(image, detections, color=(0, 255, 0), thickness=2):
    img = image.copy()

    for det in detections:
        quad = det['quad']
        points = np.array([[int(x), int(y)] for x, y in quad], dtype=np.int32)

        cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

    return img
