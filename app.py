from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import base64
import io
import pytesseract
from pytesseract import Output
import fitz  # PyMuPDF
import os
import pandas as pd
from gemini_field_extract import extract_names_from_image, prepare_gemini_output_for_matching
from ocr_match_handler import OCRMatchHandler

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="PDF & Image OCR Coordinate Extractor")

# Mount templates & static folders
templates = Jinja2Templates(directory="templates")
app.mount("/bounding_extraction/static", StaticFiles(directory="static"), name="static")

# Store OCR results temporarily
ocr_results_cache = {}


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
    """Perform OCR on image using multiple strategies to catch all text"""
    
    # Strategy 1: Use PSM 11 (Sparse text)
    custom_config = r'--oem 3 --psm 11'
    ocr1 = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config)
    
    # Strategy 2: Use PSM 6 (Uniform block of text)
    custom_config2 = r'--oem 3 --psm 6'
    ocr2 = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config2)
    
    # Strategy 3: Use PSM 12 (Sparse text with OSD)
    custom_config3 = r'--oem 3 --psm 12'
    try:
        ocr3 = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config3)
    except:
        ocr3 = {"text": [], "conf": [], "left": [], "top": [], "width": [], "height": []}
    
    # Combine all strategies
    all_boxes = {}
    
    for ocr_result in [ocr1, ocr2, ocr3]:
        for i in range(len(ocr_result["text"])):
            text = (ocr_result["text"][i] or "").strip()
            
            if not text:
                continue
                
            try:
                conf = float(ocr_result["conf"][i]) if ocr_result["conf"][i] not in ("-1", "-1.0", "", None) else 0.0
            except (ValueError, TypeError):
                conf = 0.0
                
            x = int(ocr_result["left"][i])
            y = int(ocr_result["top"][i])
            w = int(ocr_result["width"][i])
            h = int(ocr_result["height"][i])
            
            if w <= 0 or h <= 0:
                continue
            
            pos_key = (round(x / 5) * 5, round(y / 5) * 5, text.lower())
            
            if pos_key not in all_boxes or conf > all_boxes[pos_key]["conf"]:
                all_boxes[pos_key] = {
                    "x": x,
                    "y": y,
                    "right": x + w,
                    "bottom": y + h,
                    "text": text,
                    "conf": conf,
                }
    
    boxes = sorted(all_boxes.values(), key=lambda b: (b["y"], b["x"]))
    
    return boxes


def _preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess image to improve OCR accuracy"""
    from PIL import ImageEnhance
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    return img


def convert_ocr_boxes_to_structured_format(ocr_boxes, page_width, page_height, page_number=1):
    """Convert OCR boxes to structured format for OCRMatchHandler"""
    words = []
    for box in ocr_boxes:
        words.append({
            "text": box["text"],
            "x0": box["x"],
            "top": box["y"],
            "x1": box["right"],
            "bottom": box["bottom"],
        })
    
    return [{
        "page_number": page_number,
        "page_width": page_width,
        "page_height": page_height,
        "words": words
    }]


def process_names_and_match(img, page_number, ocr_boxes, w, h):
    """
    Process Gemini name extraction and match each individual name part with OCR coordinates.
    For "Amal Krishna Rajesh", finds coordinates for "Amal", "Krishna", and "Rajesh" separately.
    """
    
    # Extract names using Gemini
    gemini_result = extract_names_from_image(img)
    gemini_names = prepare_gemini_output_for_matching(gemini_result)

    # Convert OCR boxes to structured format
    structured_ocr_data = convert_ocr_boxes_to_structured_format(
        ocr_boxes, w, h, page_number
    )

    # Initialize matcher
    matcher = OCRMatchHandler(fuzzy_threshold=0.80)

    # Match each individual name part with OCR coordinates
    matched_results = []
    
    print(f"\n{'='*70}")
    print(f"MATCHING INDIVIDUAL NAME PARTS WITH OCR COORDINATES")
    print(f"{'='*70}\n")
    
    for name_obj in gemini_names:
        full_name = name_obj.get("full_name", "")
        name_parts = name_obj.get("name_parts", [])
        person_id = name_obj.get("person_id", 0)
        
        print(f"Person {person_id}: {full_name}")
        print(f"  Searching for individual parts: {name_parts}\n")
        
        # Match each individual part (e.g., "Amal", "Krishna", "Rajesh")
        for part_index, name_part in enumerate(name_parts, 1):
            if not name_part.strip():
                continue
            
            # Find match for this specific name part
            match_info, structured_ocr_data = matcher.find_best_match_for_value(
                name_part, structured_ocr_data
            )
            
            if match_info:
                matched_results.append({
                    "person_id": person_id,
                    "full_name": full_name,
                    "part_index": part_index,
                    "part_name": f"Part {part_index}",
                    "value": name_part,
                    "x": round(match_info["x0"]),
                    "y": round(match_info["top"]),
                    "right": round(match_info["x1"]),
                    "bottom": round(match_info["bottom"]),
                    "page": match_info["page_number"],
                })
                print(f"  ✓ '{name_part}' found at ({match_info['x0']}, {match_info['top']}) -> ({match_info['x1']}, {match_info['bottom']})")
            else:
                matched_results.append({
                    "person_id": person_id,
                    "full_name": full_name,
                    "part_index": part_index,
                    "part_name": f"Part {part_index}",
                    "value": name_part,
                    "x": 0, "y": 0, "right": 0, "bottom": 0,
                    "page": page_number,
                })
                print(f"  ✗ '{name_part}' NOT FOUND in OCR")
        
        print()
    
    print(f"{'='*70}\n")
    print(f"Total name parts matched: {len([m for m in matched_results if m['right'] > 0])}/{len(matched_results)}\n")
    
    return gemini_names, matched_results


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Display upload form"""
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    """Extract bounding boxes and text from uploaded PDF or image"""
    content = await file.read()
    filename = file.filename.lower()
    results = []

    if filename.endswith(".pdf"):
        doc = fitz.open(stream=content, filetype="pdf")
        for page in doc:
            img = _pil_image_from_pdf_page(page, dpi=300)
            img = _preprocess_image(img)
            
            ocr_boxes = _extract_boxes_from_image(img)
            w, h = img.size

            gemini_names, matched_results = process_names_and_match(
                img, page.number + 1, ocr_boxes, w, h
            )

            results.append({
                "width": w,
                "height": h,
                "image_b64": _image_to_base64_png(img),
                "ocr_boxes": ocr_boxes,
                "gemini_names": gemini_names,
                "matched_names": matched_results,
                "total_names": len(gemini_names),
                "page_label": f"Page {page.number + 1}",
            })
    else:
        img = Image.open(io.BytesIO(content)).convert("RGB")
        img = _preprocess_image(img)
        
        ocr_boxes = _extract_boxes_from_image(img)
        w, h = img.size

        gemini_names, matched_results = process_names_and_match(
            img, 1, ocr_boxes, w, h
        )

        results.append({
            "width": w,
            "height": h,
            "image_b64": _image_to_base64_png(img),
            "ocr_boxes": ocr_boxes,
            "gemini_names": gemini_names,
            "matched_names": matched_results,
            "total_names": len(gemini_names),
            "page_label": f"Image File: {file.filename}",
        })

    # Store results in cache for download
    ocr_results_cache['latest'] = results

    return templates.TemplateResponse("index.html", {"request": request, "results": results})


@app.get("/download-ocr-excel")
async def download_ocr_excel():
    """Download OCR coordinates as Excel file"""
    if 'latest' not in ocr_results_cache:
        return HTMLResponse(content="No OCR results available. Please upload a file first.", status_code=400)
    
    results = ocr_results_cache['latest']
    
    # Prepare data for Excel
    all_data = []
    
    for result in results:
        page_label = result['page_label']
        
        for box in result['ocr_boxes']:
            all_data.append({
                'Page': page_label,
                'Text': box['text'],
                'X': box['x'],
                'Y': box['y'],
                'Right': box['right'],
                'Bottom': box['bottom'],
                'Width': box['right'] - box['x'],
                'Height': box['bottom'] - box['y'],
                'Confidence': round(box['conf'], 2)
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='OCR Coordinates', index=False)
        
        # Auto-adjust column widths
        worksheet = writer.sheets['OCR Coordinates']
        for i, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_length)
    
    output.seek(0)
    
    return StreamingResponse(
        output,
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition': 'attachment; filename=ocr_coordinates.xlsx'}
    )