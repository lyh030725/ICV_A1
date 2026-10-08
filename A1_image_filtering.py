"""ICV Assignment #1 - Part #1. Image Filtering

Implements cross-correlation with 1D/2D kernels and the Gaussian filter
from scratch (no built-in correlation / convolution / filtering / padding).

Running this file produces the results required by 1-2 (d) and 1-2 (e)
for 'lenna.png' and 'shapes.png'.
"""

import os
import time

import cv2
import numpy as np

# Set ICV_NO_SHOW=1 to skip the GUI windows (useful for headless runs).
SHOW_WINDOWS = os.environ.get('ICV_NO_SHOW') != '1'

IMAGE_DIRS = ['.', 'A1_Images', './A1_Images', '../A1_Images']
RESULT_DIR = './result'


# ----------------------------------------------------------------------
# 1-1. Image filtering by cross-correlation
# ----------------------------------------------------------------------
def _pad_replicate(img, pad_h, pad_w):
    """Pad an image so that outside pixels copy the nearest inside pixel.

    Implemented with clamped index arrays instead of a built-in padding
    routine, as required by 1-1 (e).
    """
    h, w = img.shape[:2]
    rows = np.arange(-pad_h, h + pad_h)
    cols = np.arange(-pad_w, w + pad_w)
    rows[rows < 0] = 0
    rows[rows > h - 1] = h - 1
    cols[cols < 0] = 0
    cols[cols > w - 1] = w - 1
    return img[rows, :][:, cols]


def cross_correlation_1d(img, kernel):
    """Cross-correlate `img` with a 1D `kernel`.

    The kernel orientation is decided by its shape: an (n, 1) kernel is
    vertical, a (1, n) kernel (or a flat (n,) kernel) is horizontal.
    The output has exactly the same size as the input.
    """
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if kernel.ndim == 1:
        is_vertical = False
        taps = kernel
    elif kernel.ndim == 2 and kernel.shape[1] == 1:
        is_vertical = True
        taps = kernel[:, 0]
    elif kernel.ndim == 2 and kernel.shape[0] == 1:
        is_vertical = False
        taps = kernel[0, :]
    else:
        raise ValueError('cross_correlation_1d expects a 1D kernel, got shape %s'
                         % (kernel.shape,))

    k = taps.shape[0]
    pad = k // 2
    h, w = img.shape
    filtered_img = np.zeros((h, w), dtype=np.float64)

    if is_vertical:
        padded = _pad_replicate(img, pad, 0)
        for i in range(k):
            filtered_img += taps[i] * padded[i:i + h, :]
    else:
        padded = _pad_replicate(img, 0, pad)
        for j in range(k):
            filtered_img += taps[j] * padded[:, j:j + w]

    return filtered_img


def cross_correlation_2d(img, kernel):
    """Cross-correlate `img` with a 2D `kernel`, preserving the image size."""
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError('cross_correlation_2d expects a 2D kernel, got shape %s'
                         % (kernel.shape,))

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    h, w = img.shape

    padded = _pad_replicate(img, pad_h, pad_w)
    filtered_img = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            coeff = kernel[i, j]
            if coeff != 0.0:
                filtered_img += coeff * padded[i:i + h, j:j + w]

    return filtered_img


# ----------------------------------------------------------------------
# 1-2. The Gaussian filter
# ----------------------------------------------------------------------
def get_gaussian_filter_1d(size, sigma):
    """1D Gaussian correlation kernel of length `size` (odd), normalized to 1."""
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def get_gaussian_filter_2d(size, sigma):
    """2D Gaussian correlation kernel of shape (size, size), normalized to 1."""
    half = size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


# ----------------------------------------------------------------------
# Shared helpers, also used by Part #2 and Part #3
# ----------------------------------------------------------------------
def load_gray_image(file_name):
    """Read `file_name` as grayscale, looking in the usual image folders."""
    for directory in IMAGE_DIRS:
        path = os.path.join(directory, file_name)
        if os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    raise FileNotFoundError('Could not find image "%s" in %s' % (file_name, IMAGE_DIRS))


def ensure_result_dir():
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    return RESULT_DIR


def to_uint8(img):
    """Clip a float image to [0, 255] and cast it to uint8 for display/saving."""
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def show_image(window_name, img):
    if not SHOW_WINDOWS:
        return
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def put_caption(img, text):
    """Draw a readable caption on the top-left corner of a grayscale image.

    The font is scaled to the image width, and a dark outline is drawn under
    the white glyphs so the caption stays legible over any background.
    """
    out = img.copy()
    scale = max(0.5, img.shape[1] / 640.0)
    thickness = max(1, int(round(scale)))
    origin = (int(8 * scale), int(26 * scale))
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, 0,
                thickness + 2, cv2.LINE_AA)
    cv2.putText(out, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, 255,
                thickness, cv2.LINE_AA)
    return out


def gaussian_filter_2d(img, size, sigma):
    """Gaussian filtering with a single 2D kernel (uses 1-1)."""
    return cross_correlation_2d(img, get_gaussian_filter_2d(size, sigma))


def gaussian_filter_1d_separable(img, size, sigma):
    """Gaussian filtering by applying vertical then horizontal 1D kernels (uses 1-1)."""
    kernel = get_gaussian_filter_1d(size, sigma)
    vertical = kernel.reshape(-1, 1)
    horizontal = kernel.reshape(1, -1)
    return cross_correlation_1d(cross_correlation_1d(img, vertical), horizontal)


# ----------------------------------------------------------------------
# Part #1 report routines
# ----------------------------------------------------------------------
KERNEL_SIZES = [5, 11, 17]
SIGMAS = [1, 6, 11]


def report_gaussian_grid(img, image_name):
    """1-2 (d): 9 Gaussian filterings shown in one window and saved as one file."""
    rows = []
    for size in KERNEL_SIZES:
        row = []
        for sigma in SIGMAS:
            filtered = to_uint8(gaussian_filter_2d(img, size, sigma))
            row.append(put_caption(filtered, '%dx%d s=%d' % (size, size, sigma)))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    out_path = os.path.join(ensure_result_dir(),
                            'part_1_gaussian_filtered_%s' % image_name)
    cv2.imwrite(out_path, grid)
    print('  saved %s' % out_path)
    show_image('part_1_gaussian_filtered_%s' % image_name, grid)


def report_1d_vs_2d(img, image_name, size=17, sigma=6):
    """1-2 (e): compare the separable 1D filtering against the 2D filtering."""
    start = time.time()
    result_1d = gaussian_filter_1d_separable(img, size, sigma)
    time_1d = time.time() - start

    start = time.time()
    result_2d = gaussian_filter_2d(img, size, sigma)
    time_2d = time.time() - start

    diff = np.abs(result_1d - result_2d)

    print('  [%s] Gaussian %dx%d sigma=%d' % (image_name, size, size, sigma))
    print('    computational time (1D vertical + horizontal) : %.6f sec' % time_1d)
    print('    computational time (2D kernel)                : %.6f sec' % time_2d)
    print('    sum of absolute intensity differences         : %.9f' % diff.sum())
    print('    maximum absolute intensity difference         : %.9f' % diff.max())

    show_image('part_1_1d_vs_2d_difference_map_%s' % image_name, to_uint8(diff))


def main():
    print('1-2 (c) get_gaussian_filter_1d(5, 1) =')
    print(get_gaussian_filter_1d(5, 1))
    print('1-2 (c) get_gaussian_filter_2d(5, 1) =')
    print(get_gaussian_filter_2d(5, 1))
    print()

    for image_name in ['lenna.png', 'shapes.png']:
        img = load_gray_image(image_name)
        print('--- %s (%d x %d) ---' % (image_name, img.shape[1], img.shape[0]))
        report_gaussian_grid(img, image_name)
        report_1d_vs_2d(img, image_name)
        print()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
