from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import easyocr
import io


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_headers = ["*"],
    allow_methods = ["*"],
)

yolo_stream = None 
enhanced_stream = None

yolo_model = None
easyocr_reader = None

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("model/best_ele_robo.pt")
    return yolo_model

def get_easyocr_reader():
    global easyocr_reader
    if easyocr_reader is None:
        easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return easyocr_reader


def enhance_display_image(image):
    """Enhance the display of an image for better OCR results."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(scaled, 9, 75, 75)
    contrast = cv2.convertScaleAbs(denoised, alpha=1.3, beta=10)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    kernel = np.ones((2, 2), np.uint8)
    final = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return final

def process_image(image):
    """Processes an image using YOLO and OCR. Returns YOLO-processed image, enhanced image, and recognized text."""
    model = get_yolo_model()
    reader = get_easyocr_reader()

    results = model(image)
    detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() > 0 else []

    recognized_text = []
    enhanced_image = None

    original_image = image.copy()
    classes = ["Meter Reading"]
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        label = f"Class: {classes[int(class_id)]}: {confidence:.2f}"
        cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(original_image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
        enhanced_image = enhance_display_image(cropped_image)

        result = reader.readtext(enhanced_image, allowlist="0123456789")
        recognized_text.extend([entry[1] for entry in result])

    return original_image, enhanced_image, " ".join(recognized_text)

@app.get("/")
async def health_check():
    return {"status": "healthy"}


@app.post("/ocr")
async def recognize_text(file: UploadFile = File(...)):
    """Endpoint to recognize text in an image using YOLO for localization and EasyOCR for text extraction."""

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload JPEG or PNG images only.")

    image_bytes = await file.read()
    image = np.array(Image.open(io.BytesIO(image_bytes)))

    try:
        model = get_yolo_model()
        reader = get_easyocr_reader()

        results = model(image)
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data.numel() > 0 else []

        recognized_text = []

        for detection in detections:
            x1, y1, x2, y2, _, _ = detection

            cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

            enhanced_image = enhance_display_image(cropped_image)

            result = reader.readtext(enhanced_image, allowlist="0123456789")
            recognized_text.extend([entry[1] for entry in result])

        return {
            "recognized_text": " ".join(recognized_text),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/process")
async def process_image_endpoint(file: UploadFile = File(...)):

    """Endpoint to process an image, returning YOLO-processed image, enhanced image, and recognized text."""

    global yolo_stream, enhanced_stream

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Upload JPEG or PNG images only.")


    image_bytes = await file.read()
    image = np.array(Image.open(io.BytesIO(image_bytes)))

    try:
        yolo_image, enhanced_image, recognized_text = process_image(image)

        _, yolo_buffer = cv2.imencode(".png", yolo_image)
        yolo_stream = io.BytesIO(yolo_buffer.tobytes())

        _, enhanced_buffer = cv2.imencode(".png", enhanced_image)
        enhanced_stream = io.BytesIO(enhanced_buffer.tobytes())

        return {
            "recognized_text": recognized_text,
            "yolo_processed_image_url": "/yolo_image", 
            "enhanced_image_url": "/enhanced_image",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/yolo_image")
async def get_yolo_image():
    """Endpoint to serve the YOLO processed image"""
    if not yolo_stream:
        raise HTTPException(status_code=404, detail="YOLO image not found")
    return StreamingResponse(yolo_stream, media_type="image/png")

@app.get("/enhanced_image")
async def get_enhanced_image():
    """Endpoint to serve the enhanced image"""
    if not enhanced_stream:
        raise HTTPException(status_code=404, detail="Enhanced image not found")
    return StreamingResponse(enhanced_stream, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)