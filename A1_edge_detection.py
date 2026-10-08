"""ICV Assignment #1 - Part #2. Edge Detection

Gaussian smoothing -> Sobel image gradient -> direction-based NMS.
All filtering goes through the cross-correlation functions of Part #1.
"""

import os
import time

import cv2
import numpy as np

from A1_image_filtering import (cross_correlation_2d, ensure_result_dir,
                                gaussian_filter_2d, load_gray_image, show_image,
                                to_uint8)

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float64)

# Representative angles of the 8 quantization bins, and the (row, col) offset
# of the neighbour lying along each direction.  Angles follow the convention of
# the assignment figure (0 deg points right, 90 deg points up on screen), so the
# row offset is negated: a smaller row index is higher on screen.
QUANTIZED_ANGLES = np.arange(0, 360, 45, dtype=np.float64)
DIRECTION_OFFSETS = [(-int(round(np.sin(np.deg2rad(a)))),
                      int(round(np.cos(np.deg2rad(a)))))
                     for a in QUANTIZED_ANGLES]


# ----------------------------------------------------------------------
# 2-2. Image gradient
# ----------------------------------------------------------------------
def compute_image_gradient(img):
    """Return the magnitude and direction (degrees, [0, 360)) of the gradient."""
    dx = cross_correlation_2d(img, SOBEL_X)
    dy = cross_correlation_2d(img, SOBEL_Y)

    mag = np.sqrt(dx ** 2 + dy ** 2)
    # -dy puts the angles in the screen convention used by the assignment figure.
    dir = np.rad2deg(np.arctan2(-dy, dx)) % 360.0

    return mag, dir


# ----------------------------------------------------------------------
# 2-3. Non-maximum suppression along the gradient direction
# ----------------------------------------------------------------------
def non_maximum_suppression_dir(mag, dir):
    """Suppress every magnitude that is not a strict maximum along `dir`.

    The directions are quantized into the 8 bins [0, 45, ..., 315] degrees and
    the centre magnitude is compared against its two neighbours lying on that
    direction (no interpolation).
    """
    mag = np.asarray(mag, dtype=np.float64)
    h, w = mag.shape

    # Nearest representative angle, as a bin index in [0, 8).
    bins = (np.rint(dir / 45.0).astype(np.int32)) % 8

    # Border of zeros so that neighbours outside the image never win.
    padded = np.zeros((h + 2, w + 2), dtype=np.float64)
    padded[1:1 + h, 1:1 + w] = mag

    suppressed_mag = np.zeros((h, w), dtype=np.float64)
    for b, (dr, dc) in enumerate(DIRECTION_OFFSETS):
        selected = bins == b
        if not selected.any():
            continue
        forward = padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        backward = padded[1 - dr:1 - dr + h, 1 - dc:1 - dc + w]
        keep = selected & (mag > forward) & (mag > backward)
        suppressed_mag[keep] = mag[keep]

    return suppressed_mag


# ----------------------------------------------------------------------
# Script
# ----------------------------------------------------------------------
def process(image_name):
    img = load_gray_image(image_name)
    print('--- %s (%d x %d) ---' % (image_name, img.shape[1], img.shape[0]))

    # 2-1. Gaussian filtering with the (7, 1.5) filter of Part #1.
    smoothed = gaussian_filter_2d(img, 7, 1.5)

    # 2-2. Image gradient.
    start = time.time()
    mag, dir = compute_image_gradient(smoothed)
    print('  compute_image_gradient          : %.6f sec' % (time.time() - start))

    raw_path = os.path.join(ensure_result_dir(), 'part_2_edge_raw_%s' % image_name)
    cv2.imwrite(raw_path, to_uint8(mag))
    print('  saved %s' % raw_path)
    show_image('part_2_edge_raw_%s' % image_name, to_uint8(mag))

    # 2-3. Non-maximum suppression.
    start = time.time()
    suppressed_mag = non_maximum_suppression_dir(mag, dir)
    print('  non_maximum_suppression_dir     : %.6f sec' % (time.time() - start))

    sup_path = os.path.join(ensure_result_dir(), 'part_2_edge_sup_%s' % image_name)
    cv2.imwrite(sup_path, to_uint8(suppressed_mag))
    print('  saved %s' % sup_path)
    show_image('part_2_edge_sup_%s' % image_name, to_uint8(suppressed_mag))
    print()


def main():
    for image_name in ['shapes.png', 'lenna.png']:
        process(image_name)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
