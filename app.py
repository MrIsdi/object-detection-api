from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import base64
from io import BytesIO

app = FastAPI()
model = YOLO("yolov8n.pt")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect/")
async def detect_objects(file: UploadFile):
 # Process the uploaded image for object detection
 image_bytes = await file.read()
 image = np.frombuffer(image_bytes, dtype=np.uint8)
 image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
 # Perform object detection with YOLOv8
 detections = model(image)
 
 return detections[0].tojson()

class ImageData(BaseModel):
    image: str  # Data gambar dalam format base64
    
@app.post("/uploadimage")
async def upload_image(image_data: ImageData):
    # Mengonversi base64 ke gambar
    base64_data = image_data.image.split(',')[1]
    image = Image.open(BytesIO(base64.b64decode(base64_data)))
    detections = model(image)
    
    return detections[0].tojson()
    