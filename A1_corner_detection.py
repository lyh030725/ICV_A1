"""ICV Assignment #1 - Part #3. Corner Detection

Gaussian smoothing -> Harris corner response -> thresholding and
window-based non-maximum suppression.  All filtering goes through the
cross-correlation functions of Part #1.
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

WINDOW_SIZE = 5       # uniform window used for the second moment matrix
KAPPA = 0.04          # kappa of the Harris response function
THRESHOLD = 0.1       # response threshold of 3-3
NMS_WIN_SIZE = 11     # winSize of non_maximum_suppression_win
GREEN = (0, 255, 0)


# ----------------------------------------------------------------------
# 3-2. Corner response
# ----------------------------------------------------------------------
def compute_corner_response(img):
    """Harris corner response, clamped at 0 and normalized to [0, 1]."""
    dx = cross_correlation_2d(img, SOBEL_X)
    dy = cross_correlation_2d(img, SOBEL_Y)

    # Uniform (box) window: w(x, y) = 1 inside the window, 0 outside.
    window = np.ones((WINDOW_SIZE, WINDOW_SIZE), dtype=np.float64)
    sum_xx = cross_correlation_2d(dx * dx, window)
    sum_yy = cross_correlation_2d(dy * dy, window)
    sum_xy = cross_correlation_2d(dx * dy, window)

    # R = l1*l2 - k*(l1+l2)^2 = det(M) - k*trace(M)^2
    det = sum_xx * sum_yy - sum_xy ** 2
    trace = sum_xx + sum_yy
    R = det - KAPPA * trace ** 2

    R[R < 0] = 0.0
    peak = R.max()
    if peak > 0:
        R = R / peak

    return R


# ----------------------------------------------------------------------
# 3-3. Window-based non-maximum suppression
# ----------------------------------------------------------------------
def non_maximum_suppression_win(R, winSize):
    """Keep a response only if it is the maximum inside a winSize x winSize
    window centred on it and greater than the threshold 0.1."""
    R = np.asarray(R, dtype=np.float64)
    h, w = R.shape
    half = winSize // 2

    # Border of -inf so that positions outside the image never win the maximum.
    padded = np.full((h + 2 * half, w + 2 * half), -np.inf, dtype=np.float64)
    padded[half:half + h, half:half + w] = R

    # The maximum over a square window is separable: rows first, then columns.
    row_max = np.full((h, w + 2 * half), -np.inf, dtype=np.float64)
    for d in range(winSize):
        row_max = np.maximum(row_max, padded[d:d + h, :])

    win_max = np.full((h, w), -np.inf, dtype=np.float64)
    for d in range(winSize):
        win_max = np.maximum(win_max, row_max[:, d:d + w])

    suppressed_R = np.zeros((h, w), dtype=np.float64)
    keep = (R >= win_max) & (R > THRESHOLD)
    suppressed_R[keep] = R[keep]

    return suppressed_R


# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------
def draw_thresholded_corners(img, R):
    """3-3 (a): paint every pixel whose response exceeds the threshold green.

    The overlay is drawn on the original image so the corners stay easy to read.
    """
    out = cv2.cvtColor(to_uint8(img), cv2.COLOR_GRAY2BGR)
    out[R > THRESHOLD] = GREEN
    return out


def draw_corner_circles(img, suppressed_R, radius=5):
    """3-3 (d): draw a green circle on every surviving local maximum."""
    out = cv2.cvtColor(to_uint8(img), cv2.COLOR_GRAY2BGR)
    for y, x in zip(*np.nonzero(suppressed_R)):
        cv2.circle(out, (int(x), int(y)), radius, GREEN, 2)
    return out


# ----------------------------------------------------------------------
# Script
# ----------------------------------------------------------------------
def process(image_name):
    img = load_gray_image(image_name)
    print('--- %s (%d x %d) ---' % (image_name, img.shape[1], img.shape[0]))

    # 3-1. Gaussian filtering with the (7, 1.5) filter of Part #1.
    smoothed = gaussian_filter_2d(img, 7, 1.5)

    # 3-2. Corner response.
    start = time.time()
    R = compute_corner_response(smoothed)
    print('  compute_corner_response         : %.6f sec' % (time.time() - start))

    raw_path = os.path.join(ensure_result_dir(), 'part_3_corner_raw_%s' % image_name)
    cv2.imwrite(raw_path, to_uint8(R * 255.0))
    print('  saved %s' % raw_path)
    show_image('part_3_corner_raw_%s' % image_name, to_uint8(R * 255.0))

    # 3-3 (a, b). Thresholding.
    bin_img = draw_thresholded_corners(img, R)
    bin_path = os.path.join(ensure_result_dir(), 'part_3_corner_bin_%s' % image_name)
    cv2.imwrite(bin_path, bin_img)
    print('  saved %s' % bin_path)
    show_image('part_3_corner_bin_%s' % image_name, bin_img)

    # 3-3 (c, d). Window-based NMS.
    start = time.time()
    suppressed_R = non_maximum_suppression_win(R, NMS_WIN_SIZE)
    print('  non_maximum_suppression_win     : %.6f sec' % (time.time() - start))
    print('  detected corners                : %d' % np.count_nonzero(suppressed_R))

    sup_img = draw_corner_circles(img, suppressed_R)
    sup_path = os.path.join(ensure_result_dir(), 'part_3_corner_sup_%s' % image_name)
    cv2.imwrite(sup_path, sup_img)
    print('  saved %s' % sup_path)
    show_image('part_3_corner_sup_%s' % image_name, sup_img)
    print()


def main():
    for image_name in ['shapes.png', 'lenna.png']:
        process(image_name)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
