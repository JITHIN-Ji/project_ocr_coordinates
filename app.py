from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import base64
import io
import pytesseract
from pytesseract import Output
import fitz  # PyMuPDF

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="PDF & Image OCR Coordinate Extractor")

# Mount templates & static folders
templates = Jinja2Templates(directory="templates")
app.mount("/bounding_extraction/static", StaticFiles(directory="static"), name="static")


def _pil_image_from_pdf_page(pdf_page: fitz.Page, dpi: int = 200) -> Image.Image:
    """Convert a PDF page to a PIL image"""
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = pdf_page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return image.convert("RGB")


def _image_to_base64_png(image: Image.Image) -> str:
    """Convert PIL image to base64 PNG"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _extract_boxes_from_image(img: Image.Image):
    """Perform OCR on image and return bounding boxes + coordinates"""
    ocr = pytesseract.image_to_data(img, output_type=Output.DICT)
    boxes = []
    for i in range(len(ocr["text"])):
        text = (ocr["text"][i] or "").strip()
        conf = float(ocr["conf"][i]) if ocr["conf"][i] not in ("-1", "-1.0", "", None) else -1.0
        x, y, w, h = int(ocr["left"][i]), int(ocr["top"][i]), int(ocr["width"][i]), int(ocr["height"][i])
        if text and conf >= 0:
            boxes.append({
                "x": x,
                "y": y,
                "right": x + w,
                "bottom": y + h,
                "text": text,
                "conf": conf,
            })
    return boxes


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Display upload form"""
    return templates.TemplateResponse("index.html", {"request": request, "pages": None})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    """Extract bounding boxes and text from uploaded PDF or image"""
    content = await file.read()
    filename = file.filename.lower()
    pages_payload = []

    if filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            img = _pil_image_from_pdf_page(page, dpi=200)
            boxes = _extract_boxes_from_image(img)
            w, h = img.size

            pages_payload.append({
                "width": w,
                "height": h,
                "image_b64": _image_to_base64_png(img),
                "boxes": boxes,
                "page_label": f"Page {page.number + 1}",
            })
    else:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        boxes = _extract_boxes_from_image(img)
        w, h = img.size

        pages_payload.append({
            "width": w,
            "height": h,
            "image_b64": _image_to_base64_png(img),
            "boxes": boxes,
            "page_label": f"Image File: {file.filename}",
        })

    return templates.TemplateResponse("index.html", {"request": request, "pages": pages_payload})
