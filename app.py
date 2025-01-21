from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import easyocr
import uuid

# Create FastAPI app
app = FastAPI()

# Define folders
UPLOAD_FOLDER = "uploads"
CROPPED_FOLDER = "cropped"
PROCESSED_FOLDER = "processed"

# Mount static files for processed images
app.mount("/processed_images", StaticFiles(directory=PROCESSED_FOLDER), name="processed_images")

# Ensure directories exist
for folder in [UPLOAD_FOLDER, CROPPED_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Load YOLO model and EasyOCR reader
MODEL_PATH = "model/best_ele_robo.pt"
yolo_model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)


def enhance_display_image(image_path):
    """
    Enhances the display of an image for better OCR results.
    """
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Resize for better visibility
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Denoise the image
    denoised = cv2.bilateralFilter(scaled, 9, 75, 75)

    # Enhance contrast
    contrast = cv2.convertScaleAbs(denoised, alpha=1.3, beta=10)

    # Apply thresholding
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Sharpen the image
    kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # Apply morphological closing to enhance structure
    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return final


def process_image(image_path):
    """
    Processes an image using YOLO, applies CV operations, and performs OCR.
    Returns bounding box results, enhanced image path, and recognized text.
    """
    # Perform YOLO inference
    results = yolo_model(image_path)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() > 0 else []

    bounding_boxes = []
    recognized_text = []
    enhanced_image_path = None

    for detection in detections:
        # YOLO detection format: [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2, confidence, class_id = detection

        # Open the image
        image = Image.open(image_path)

        # Crop the detected region
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        cropped_image = image.crop(bbox)

        # Save cropped image
        cropped_image_path = os.path.join(CROPPED_FOLDER, f"{uuid.uuid4().hex}_cropped.png")
        cropped_image.save(cropped_image_path)

        # Enhance the cropped image
        enhanced_image = enhance_display_image(cropped_image_path)

        # Save the processed image
        enhanced_image_path = os.path.join(PROCESSED_FOLDER, f"{uuid.uuid4().hex}_processed.png")
        cv2.imwrite(enhanced_image_path, enhanced_image)

        # Perform OCR on enhanced image
        result = reader.readtext(enhanced_image, allowlist='0123456789')

        # Extract text from OCR results
        for entry in result:
            bbox, text, prob = entry
            recognized_text.append(text)

        # Append bounding box details
        bounding_boxes.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "confidence": float(confidence),
            "class_id": int(class_id)
        })

    return bounding_boxes, enhanced_image_path, " ".join(recognized_text)


@app.post("/ocr/")
async def recognize_text(file: UploadFile = File(...)):
    """
    Endpoint to process an image and return recognized text.
    """
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload JPEG or PNG images only.")

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image
    _, _, recognized_text = process_image(file_path)

    return JSONResponse({"recognized_text": recognized_text})


@app.post("/process/")
async def process_image_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to process an image, return bounding boxes, processed image, and recognized text.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload JPEG/PNG/JPG images only.")

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image
    bounding_boxes, enhanced_image_path, recognized_text = process_image(file_path)

    return JSONResponse({
        "bounding_boxes": bounding_boxes,
        "processed_image_url": f"/processed_images/{os.path.basename(enhanced_image_path)}",
        "recognized_text": recognized_text
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
