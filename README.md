# Bounding Extraction (PDF OCR with Bounding Boxes)

This is a minimal, standalone FastAPI app that lets you upload a PDF, runs OCR on each page, and shows the extracted text with bounding boxes drawn on top of the page images.

It does only this and nothing else.

## How to run

Prereqs (already listed in the project requirements):
- PyMuPDF (`fitz`) for rendering PDF pages to images
- Pillow for image handling
- FastAPI + Uvicorn for the web app
- pytesseract for OCR (Python bindings)
- System Tesseract OCR binary installed and available on PATH (Windows: install Tesseract and optionally set `pytesseract.pytesseract.tesseract_cmd`)

Steps:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure Tesseract OCR is installed on your system.
   - Windows: Install from `https://github.com/UB-Mannheim/tesseract/wiki`
   - If not in PATH, set the binary path in `app.py` before use, e.g.:
     ```python
     import pytesseract
     pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
     ```
3. Run the app from the repository root:
   ```bash
   uvicorn bounding_extraction.app:app --reload
   ```
4. Open `http://127.0.0.1:8000/` in your browser, upload a PDF, and view bounding boxes.

## How it works

- PDF to image: Each PDF page is rendered to a bitmap using PyMuPDF at 200 DPI.
- OCR: We pass the page image to Tesseract via `pytesseract.image_to_data`, which returns a list of text items with bounding boxes (left, top, width, height) and confidence.
- Filtering: We keep items with non-empty text and non-negative confidence.
- Display: The page image is embedded as a base64 PNG, and bounding boxes are absolutely positioned `<div>` overlays to visualize detected text regions with labels.

File overview:
- `app.py` — FastAPI app with two routes: `/` (upload form) and `/upload` (OCR + display).
- `templates/index.html` — Minimal HTML that shows the uploaded pages and overlays boxes.

## Notes
- If you need higher accuracy, increase DPI (e.g., 300) in `_pil_image_from_pdf_page`.
- Very large PDFs may be slow to render and OCR; consider limiting pages if needed.
- If you see no boxes, verify Tesseract is installed and detectable by `pytesseract`.



