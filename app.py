from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import easyocr

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with a specific origin like "http://localhost:5500" for better security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Define folders with environment support
BASE_DIR = os.getenv("BASE_DIR", ".")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
CROPPED_FOLDER = os.path.join(BASE_DIR, "cropped")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

# Mount static files with cache control
app.mount(
    "/processed_images",
    StaticFiles(directory=PROCESSED_FOLDER, html=True),
    name="processed_images",
)

# Ensure directories exist
for folder in [UPLOAD_FOLDER, CROPPED_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables for YOLO and EasyOCR
yolo_model = None
reader = None


def get_yolo_model():
    """Lazy load the YOLO model."""
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("model/best_ele_robo.pt")
    return yolo_model


def get_easyocr_reader():
    """Lazy load the EasyOCR reader."""
    global reader
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False)
    return reader


def enhance_display_image(image_path):
    """Enhance the display of an image for better OCR results."""
    img = cv2.imread(image_path)

    # Convert to grayscale if not already
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Resize, denoise, and enhance contrast
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(scaled, 9, 75, 75)
    contrast = cv2.convertScaleAbs(denoised, alpha=1.3, beta=10)

    # Threshold and sharpen
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # Morphological operations
    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel, iterations=1)

    return final


def process_image(image_path):
    """Processes an image using YOLO and OCR. Returns bounding boxes, YOLO-processed image path, enhanced image path, and recognized text."""
    try:
        # Get models
        model = get_yolo_model()
        reader = get_easyocr_reader()

        # Perform YOLO inference
        results = model(image_path)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() > 0 else []

        bounding_boxes = []
        recognized_text = []
        enhanced_image_path = None

        # Load original image for bounding box overlay
        original_image = cv2.imread(image_path)

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection

            # Draw bounding boxes and labels on the original image
            label = f"Class {int(class_id)}: {confidence:.2f}"
            cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(original_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Open and crop image
            image = Image.open(image_path)
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            cropped_image = image.crop(bbox)

            # Save cropped image
            cropped_image_path = os.path.join(CROPPED_FOLDER, f"{uuid.uuid4().hex}_cropped.png")
            cropped_image.save(cropped_image_path)

            # Enhance cropped image
            enhanced_image = enhance_display_image(cropped_image_path)

            # Save enhanced image
            enhanced_image_path = os.path.join(PROCESSED_FOLDER, f"{uuid.uuid4().hex}_processed.png")
            cv2.imwrite(enhanced_image_path, enhanced_image)

            # Perform OCR
            result = reader.readtext(enhanced_image, allowlist="0123456789")
            recognized_text.extend([entry[1] for entry in result])

            # Record bounding box details
            bounding_boxes.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": float(confidence),
                "class_id": int(class_id),
            })

        # Save YOLO-processed image
        yolo_processed_image_path = os.path.join(PROCESSED_FOLDER, f"{uuid.uuid4().hex}_yolo_processed.png")
        cv2.imwrite(yolo_processed_image_path, original_image)

        return bounding_boxes, yolo_processed_image_path, enhanced_image_path, " ".join(recognized_text)

    finally:
        # Clean up temporary files
        if os.path.exists(image_path):
            os.remove(image_path)



@app.post("/ocr/")
async def recognize_text(file: UploadFile = File(...)):
    """Endpoint to recognize text in an image."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload JPEG or PNG images only.")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process image
    _, _, _, recognized_text = process_image(file_path)
    return JSONResponse({"recognized_text": recognized_text})


@app.post("/process/")
async def process_image_endpoint(file: UploadFile = File(...)):
    """Endpoint to process an image, returning bounding boxes, YOLO-processed image, enhanced image, and recognized text."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload JPEG or PNG images only.")

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process image
    bounding_boxes, yolo_processed_image_path, enhanced_image_path, recognized_text = process_image(file_path)

    return JSONResponse({
        "bounding_boxes": bounding_boxes,
        "yolo_processed_image_url": f"/processed_images/{os.path.basename(yolo_processed_image_path)}",
        "enhanced_image_url": f"/processed_images/{os.path.basename(enhanced_image_path)}",
        "recognized_text": recognized_text,
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
    uvicorn.run(app, host="127.0.0.1", port=port)
