import io
import re
import cv2
import logging
import pytesseract
import numpy as np
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LPR API MVP", description="Lightweight API for License Plate Recognition")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For MVP, allow all origins. Can be restricted to Vercel later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model (License Plate Detector version)
# Ensure license_plate_detector.pt is in the same directory
model = YOLO('license_plate_detector.pt')

# Regex for Brazilian Mercosul Plate: AAA1A23
PLATE_REGEX = re.compile(r'[A-Z]{3}[0-9][A-Z][0-9]{2}')

def process_image_for_ocr(crop_img):
    """
    Apply image processing to improve Tesseract OCR accuracy.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # Resize image to make it larger for Tesseract
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Add a slight blur to remove noise
    blur = cv2.GaussianBlur(thresh, (3,3), 0)
    
    return blur

@app.post("/read-plate")
async def read_plate(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image
    contents = await file.read()
    
    # Size check (limit to ~5MB to avoid memory exhaustion on Render Free Tier)
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size exceeds 5MB limit")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    logger.info("Running YOLO inference")
    # Run YOLO inference
    results = model.predict(img, imgsz=320, conf=0.25) # Small image size to save memory/cpu

    plate_match = None
    raw_text = ""
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            # Crop the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue

            processed_crop = process_image_for_ocr(crop)
            
            # Tesseract OCR config:
            # --psm 8: Treat the image as a single word
            # --oem 3: Default OCR Engine Mode
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(processed_crop, config=custom_config)
            
            clean_text = "".join(text.split())
            raw_text += clean_text + " "

            # Search for plate regex
            match = PLATE_REGEX.search(clean_text)
            if match:
                plate_match = match.group()
                break
    else:
        # Fallback: try OCR on the whole image if YOLO finds nothing
        logger.info("No objects detected, falling back to whole image OCR")
        processed_img = process_image_for_ocr(img)
        custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        clean_text = "".join(text.split())
        raw_text += clean_text
        match = PLATE_REGEX.search(clean_text)
        if match:
            plate_match = match.group()

    if not plate_match:
        # Return what we found or null if nothing matched the regex
        return {
            "plate": None,
            "raw_text": raw_text.strip(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    return {
        "plate": plate_match,
        "raw_text": raw_text.strip(),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.get("/")
def health_check():
    return {"status": "ok", "message": "LPR API is running"}
