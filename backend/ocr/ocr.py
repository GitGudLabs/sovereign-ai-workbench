import ollama
import pymupdf as fitz
import os
import tempfile

def ocr_image(image_path: str) -> str:
    """
    Runs OCR on a single image file using qwen2.5vl via Ollama.
    """
    prompt = (
        "Transcribe all visible text in this image exactly as it appears. "
        "Rules: Include ALL printed and handwritten text. "
        "Preserve the original spelling, capitalization, numbers, punctuation, and symbols. "
        "Preserve the reading order and line breaks as closely as possible. "
        "IMPORTANT: Preserve natural word spacing exactly as a human would read it — "
        "always put a space between a number and its unit (e.g. '10,000 L', not '10,000L'), "
        "between a date's day and month and year (e.g. '12 August 2026', not '12 August2026'), "
        "between a list number and the following word (e.g. '1. Equipment', not '1.Equipment'), "
        "and between any two separate words that appear next to each other. "
        "Never merge two words or a number and a word together with no space between them. "
        "Do not describe, summarize, interpret, or explain the image. "
        "Do not correct spelling or grammar. "
        "Do not invent missing text. "
        "If any text is genuinely unreadable, write [unclear]. "
        "For tables, preserve the row and column structure as clearly as possible. "
        "Return ONLY the transcription."
    )

    try:
        response = ollama.chat(
            model="qwen2.5vl:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_path]
                }
            ],
            options={
                "num_ctx": 8192
            }
        )
        return response["message"]["content"]

    except Exception as e:
        # If the model call fails for any reason (model not found, Ollama not running,
        # context size exceeded, etc.), return a clear error message instead of crashing.
        return f"[ERROR: OCR failed for this page — {str(e)}]"


def pdf_to_images(pdf_path: str, output_folder: str) -> list:
    """
    Converts each page of a PDF into a PNG image.
    Returns a list of image file paths, one per page.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not open PDF at '{pdf_path}': {e}")

    image_paths = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        pix = page.get_pixmap(dpi=200)
        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)

    doc.close()
    return image_paths


def ocr_document(file_path: str) -> list:
    """
    Takes a path to either an image or a PDF.
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    For a plain image, this returns a single-item list (page 1).
    """
    if not os.path.exists(file_path):
        return [{"page": 1, "text": f"[ERROR: File not found — {file_path}]"}]

    file_extension = os.path.splitext(file_path)[1].lower()
    results = []

    if file_extension == ".pdf":
        try:
            with tempfile.TemporaryDirectory() as temp_folder:
                page_images = pdf_to_images(file_path, temp_folder)
                for i, page_image in enumerate(page_images):
                    print(f"Processing page {i + 1} of {len(page_images)}...")
                    page_text = ocr_image(page_image)
                    results.append({"page": i + 1, "text": page_text})
        except Exception as e:
            results.append({"page": 1, "text": f"[ERROR: Failed to process PDF — {str(e)}]"})
    else:
        # treat as a single-page image
        page_text = ocr_image(file_path)
        results.append({"page": 1, "text": page_text})

    return results


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python ocr.py <path_to_image_or_pdf>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = ocr_document(file_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))