import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Tuple


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
    """
    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while len(detections) > 0:
        current = detections.pop(0)
        keep.append(current)

        filtered = []
        for det in detections:
            iou = polygon_iou(current['quad'], det['quad'])
            if iou < iou_threshold:
                filtered.append(det)

        detections = filtered

    return keep


def scale_and_format(detections, original_size):
    orig_h, orig_w = original_size
    scale_x = orig_w / 1024
    scale_y = orig_h / 1024

    results = []
    for det in detections:
        quad = det['quad']
        scaled_quad = [[x * scale_x, y * scale_y] for x, y in quad]

        xs = [p[0] for p in scaled_quad]
        ys = [p[1] for p in scaled_quad]

        result = det.copy()

        result.update({
            'quad': scaled_quad,
            'left': min(xs) / orig_w,
            'right': max(xs) / orig_w,
            'upper': min(ys) / orig_h,
            'lower': max(ys) / orig_h
        })

        results.append(result)

    return results


def draw_boxes(image, detections, color=(0, 255, 0), thickness=2):
    img = image.copy()

    for det in detections:
        quad = det['quad']
        points = np.array([[int(x), int(y)] for x, y in quad], dtype=np.int32)

        cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

    return img


def extract_region_features(
    feature_map: np.ndarray,
    quad: np.ndarray,
    target_height: int = 8,
    target_width: int = 32
) -> np.ndarray:
    """
    Extract and rectify region features using perspective transform.

    Args:
        feature_map: Feature map [128, 256, 256]
        quad: Quadrilateral in feature space coordinates
        target_height: Target height (default: 8)
        target_width: Target width (default: 32)

    Returns:
        Rectified features [128, target_height, target_width]
    """
    C, H, W = feature_map.shape

    # Destination quad (rectangle)
    dst_quad = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_quad)

    # Warp each channel
    rectified = np.zeros((C, target_height, target_width), dtype=np.float32)
    for c in range(C):
        channel = feature_map[c]
        rectified[c] = cv2.warpPerspective(
            channel, M, (target_width, target_height), flags=cv2.INTER_LINEAR
        )

    return rectified


def extract_rectified_quads_for_relational(
    feature_map: np.ndarray,
    quad: np.ndarray,
    target_height: int = 8,
    target_width: int = 128
) -> np.ndarray:
    """
    Extract rectified quad features for relational model [128, 2, 3].

    Args:
        feature_map: Feature map [128, 256, 256]
        quad: Quadrilateral in feature space coordinates
        target_height: Target height (default: 8)
        target_width: Target width (default: 128)

    Returns:
        Sampled features [128, 2, 3]
    """
    C, H, W = feature_map.shape

    # Destination quad (rectangle)
    dst_quad = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype=np.float32)

    # Get perspective transform
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst_quad)

    # Warp to full resolution first
    rectified_full = np.zeros((C, target_height, target_width), dtype=np.float32)
    for c in range(C):
        channel = feature_map[c]
        rectified_full[c] = cv2.warpPerspective(
            channel, M, (target_width, target_height), flags=cv2.INTER_LINEAR
        )

    # Sample spatial points for relational model
    rectified_sampled = np.zeros((C, 2, 3), dtype=np.float32)
    v_pos = [target_height // 4, 3 * target_height // 4]  # [2, 6] for height=8
    h_pos = [target_width // 4, target_width // 2, 3 * target_width // 4]  # [32, 64, 96] for width=128

    for v_idx, v in enumerate(v_pos):
        for h_idx, h in enumerate(h_pos):
            rectified_sampled[:, v_idx, h_idx] = rectified_full[:, v, h]

    return rectified_sampled


def extract_text_from_detections(
    detections: List[Dict],
    detector_features: np.ndarray,
    recognizer_session: ort.InferenceSession,
    input_name: str,
    idx_to_char: Dict[int, str]
) -> List[Dict]:
    """
    Extract text from detected regions using the recognizer model.

    Args:
        detections: List of detection dicts with 'quad', 'confidence', and 'recog_features'
        detector_features: Feature map from detector [1, 128, 256, 256]
        recognizer_session: ONNX Runtime session for recognizer
        input_name: Input name for recognizer model
        idx_to_char: Character mapping

    Returns:
        List of detections with added 'text' and 'text_confidence' fields
    """
    from .util import decode_tokens_to_text

    results = []

    for i, det in enumerate(detections):
        try:
            # Get pre-extracted features
            if 'recog_features' in det:
                recog_features = det['recog_features']
            else:
                print(f"No recog_features for detection {i}")
                det_with_text = det.copy()
                det_with_text['text'] = ""
                det_with_text['text_confidence'] = 0.0
                results.append(det_with_text)
                continue

            features_batch = np.expand_dims(recog_features, axis=0)

            # Run recognizer (returns logits AND features)
            recognizer_output = recognizer_session.run(None, {input_name: features_batch})
            logits = recognizer_output[0][0]  # [seq_len, num_classes]

            token_ids = np.argmax(logits, axis=-1)  # [seq_len]

            text, _ = decode_tokens_to_text(token_ids, idx_to_char)

            # Calculate confidence
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            confidences = np.array([softmax[s, token_ids[s]] for s in range(len(token_ids))])

            # Average confidence for valid tokens (>2)
            eos_pos = np.where(token_ids == 1)[0]
            seq_len = eos_pos[0] if len(eos_pos) > 0 else len(token_ids)
            valid_mask = token_ids[:seq_len] > 2
            if np.any(valid_mask):
                text_conf = float(np.mean(confidences[:seq_len][valid_mask]))
            else:
                text_conf = 0.0

            # Extract recognizer features (output[1]) for relational model
            if len(recognizer_output) > 1:
                recog_output_features = recognizer_output[1][0]  # [seq_len, feature_dim]
            else:
                recog_output_features = None

            det_with_text = det.copy()
            det_with_text['text'] = text
            det_with_text['text_confidence'] = text_conf
            if recog_output_features is not None:
                det_with_text['recognizer_output_features'] = recog_output_features
            results.append(det_with_text)

        except Exception as e:
            print(f"Failed to extract text for detection {i}: {e}")
            # Add detection without text
            det_with_text = det.copy()
            det_with_text['text'] = ""
            det_with_text['text_confidence'] = 0.0
            results.append(det_with_text)

    return results


def visualize_detections_with_text(
    image: np.ndarray,
    detections: List[Dict],
    show_text: bool = True,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    img = image.copy()

    for det in detections:
        quad = det['quad']
        points = np.array([[int(x), int(y)] for x, y in quad], dtype=np.int32)

        cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

        if show_text and 'text' in det and det['text']:
            x, y = int(quad[0][0]), int(quad[0][1])
            text = det['text'][:20]
            conf = det.get('text_confidence', det.get('confidence', 0))
            label = f"{text} ({conf:.2f})"

            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - text_h - 5), (x + text_w, y), (0, 0, 0), -1)
            cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def process_detections_with_features(
    detector_outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
    original_size: Tuple[int, int],
    detection_threshold: float = 0.7,
    iou_threshold: float = 0.5,
    device_name: str = "CPU"
) -> List[Dict]:
    """
    Process detector outputs: extract detections, apply NMS, extract features, and scale.

    Args:
        detector_outputs: Tuple of (confidence, rboxes, features) from detector
        original_size: Original image size (h, w)
        detection_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        device_name: Name for logging (e.g., "CPU" or "NPU")

    Returns:
        List of detections with quad, confidence, recog_features, relational_quad, and normalized coords
    """
    confidence, rboxes, features = detector_outputs
    feat_map = features[0]  # [128, 256, 256]

    # Extract raw detections
    detections = extract_detections(confidence, rboxes, threshold=detection_threshold)
    print(f"  {device_name}: Found {len(detections)} raw detections")

    # Apply NMS
    detections = non_maximum_suppression(detections, iou_threshold=iou_threshold)
    print(f"  {device_name}: After NMS: {len(detections)} detections")

    # Extract features for each detection
    for det in detections:
        quad_img = det['quad']
        # Convert from 1024x1024 image space to 256x256 feature space
        quad_feat = np.array(quad_img, dtype=np.float32) / 4.0

        # Extract rectified features for recognizer
        recog_features = extract_region_features(feat_map, quad_feat, 8, 32)
        det['recog_features'] = recog_features

        # Extract rectified features for relational model
        relational_quad = extract_rectified_quads_for_relational(feat_map, quad_feat, 8, 128)
        det['relational_quad'] = relational_quad

    # Scale to original image size and add normalized coordinates
    scaled_detections = scale_and_format(detections, original_size)

    return scaled_detections


def visualize_nearest_neighbors(
    image: np.ndarray,
    detections: List[Dict],
    neighbor_indices: List[List[int]],
    max_neighbors: int = 5,
    color: Tuple[int, int, int] = (0, 255, 255)
) -> np.ndarray:
    img = image.copy()

    # Draw detections
    for det in detections:
        quad = det['quad']
        points = np.array([[int(x), int(y)] for x, y in quad], dtype=np.int32)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw neighbor connections
    for i, (det, neighbors) in enumerate(zip(detections, neighbor_indices)):
        # Get center of current detection
        quad = det['quad']
        center_x = int(np.mean([p[0] for p in quad]))
        center_y = int(np.mean([p[1] for p in quad]))

        # Draw connections to top-k neighbors
        for j, neighbor_idx in enumerate(neighbors[:max_neighbors]):
            if 0 <= neighbor_idx < len(detections) and neighbor_idx != i:
                neighbor_quad = detections[neighbor_idx]['quad']
                neighbor_x = int(np.mean([p[0] for p in neighbor_quad]))
                neighbor_y = int(np.mean([p[1] for p in neighbor_quad]))

                # Draw arrow from current to neighbor
                cv2.arrowedLine(
                    img,
                    (center_x, center_y),
                    (neighbor_x, neighbor_y),
                    color,
                    thickness=2,
                    tipLength=0.2
                )

                # Draw rank number
                mid_x = (center_x + neighbor_x) // 2
                mid_y = (center_y + neighbor_y) // 2
                cv2.putText(
                    img, str(j+1),
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2
                )

    return img
