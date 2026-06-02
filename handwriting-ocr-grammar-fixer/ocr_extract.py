"""Extract text from preprocessed handwritten images using Tesseract OCR."""

import pytesseract
import numpy as np
from PIL import Image
import cv2

# Uncomment and set path if on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text(image: np.ndarray, lang: str = 'eng') -> str:
    """
    Extract text from a preprocessed (thresholded) image array.
    Uses Tesseract with page segmentation mode 6 (uniform block of text).
    """
    pil_image = Image.fromarray(image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)
    return text.strip()


def extract_with_confidence(image: np.ndarray) -> dict:
    """Extract text along with per-word confidence scores."""
    pil_image = Image.fromarray(image)
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

    words = []
    for i, word in enumerate(data['text']):
        word = word.strip()
        if word and int(data['conf'][i]) > 0:
            words.append({'word': word, 'confidence': int(data['conf'][i])})

    full_text = ' '.join([w['word'] for w in words])
    avg_confidence = np.mean([w['confidence'] for w in words]) if words else 0

    return {
        'text': full_text,
        'avg_confidence': round(avg_confidence, 2),
        'word_details': words
    }


if __name__ == '__main__':
    import sys
    from preprocess import preprocess
    path = sys.argv[1] if len(sys.argv) > 1 else 'sample_images/sample1.jpg'
    processed = preprocess(path)
    result = extract_with_confidence(processed)
    print(f"Extracted Text:\n{result['text']}")
    print(f"\nAvg Confidence: {result['avg_confidence']}%")
