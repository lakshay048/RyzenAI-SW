import cv2
import numpy as np


def preprocess_image(image_path, target_size=1024, output_format='NCHW'):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load: {image_path}")

    print(f"Original image shape: {img.shape}")
    original_img = img.copy()
    h, w = img.shape[:2]

    img_resized = cv2.resize(img, (target_size, target_size))

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0

    if output_format == 'NCHW':
        # CHW format with batch: (1, 3, 1024, 1024)
        img_chw = np.transpose(img_norm, (2, 0, 1))
        img_batch = np.expand_dims(img_chw, axis=0)
    elif output_format == 'NHWC':
        # HWC format with batch: (1, 1024, 1024, 3)
        img_batch = np.expand_dims(img_norm, axis=0)
    else:
        raise ValueError(f"Unsupported format: {output_format}. Use 'NCHW' or 'NHWC'")

    print(f"Preprocessed input shape: {img_batch.shape} ({output_format} format)")
    print(f"Input data range: [{img_batch.min():.3f}, {img_batch.max():.3f}]")

    return img_batch, original_img, (h, w)
