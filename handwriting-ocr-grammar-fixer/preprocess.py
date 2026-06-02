"""Image preprocessing to improve OCR accuracy on handwritten images."""

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten slightly tilted text using moments."""
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def threshold(gray: np.ndarray) -> np.ndarray:
    """Adaptive thresholding for uneven lighting in handwritten images."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )


def preprocess(image_path: str) -> np.ndarray:
    """Full preprocessing pipeline: load → gray → denoise → deskew → threshold."""
    img = load_image(image_path)
    gray = to_grayscale(img)
    denoised = denoise(gray)
    deskewed = deskew(denoised)
    thresholded = threshold(deskewed)
    return thresholded


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'sample_images/sample1.jpg'
    result = preprocess(path)
    cv2.imwrite('outputs/preprocessed.png', result)
    print(f"[SAVED] Preprocessed image saved to outputs/preprocessed.png")
