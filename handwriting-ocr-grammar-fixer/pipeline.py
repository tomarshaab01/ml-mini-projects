"""Full end-to-end pipeline: image → preprocessed → OCR text → grammar fixed output."""

import argparse
import json
import os
from preprocess import preprocess
from ocr_extract import extract_text, extract_with_confidence
from grammar_fix import analyze

OS_DIR = 'outputs'
os.makedirs(OS_DIR, exist_ok=True)


def run_pipeline(image_path: str, verbose: bool = True) -> dict:
    print(f"\n{'='*55}")
    print(f" Handwriting OCR + Grammar Fixer Pipeline")
    print(f"{'='*55}")
    print(f"\n[1/3] Preprocessing image: {image_path}")
    processed_img = preprocess(image_path)

    print("[2/3] Extracting text via OCR...")
    ocr_result = extract_with_confidence(processed_img)
    raw_text = ocr_result['text']
    print(f"  Raw OCR text ({ocr_result['avg_confidence']}% avg confidence):")
    print(f"  {raw_text}\n")

    print("[3/3] Fixing grammar...")
    grammar_result = analyze(raw_text)
    corrected_text = grammar_result['corrected']
    print(f"  Corrected text:")
    print(f"  {corrected_text}")

    if grammar_result['num_corrections'] > 0:
        print(f"\n  • {grammar_result['num_corrections']} correction(s) made:")
        for c in grammar_result['corrections']:
            print(f"    '{c['error']}' → {c['suggestions'][:1]}")

    result = {
        'image': image_path,
        'raw_text': raw_text,
        'ocr_confidence': ocr_result['avg_confidence'],
        'corrected_text': corrected_text,
        'num_corrections': grammar_result['num_corrections'],
        'corrections': grammar_result['corrections']
    }

    # Save output
    out_file = os.path.join(OS_DIR, 'result.json')
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[SAVED] Full result saved to '{out_file}'")
    print('='*55)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Handwriting OCR + Grammar Fixer')
    parser.add_argument('--image', type=str, default='sample_images/sample1.jpg',
                        help='Path to handwritten image')
    args = parser.parse_args()
    run_pipeline(args.image)
