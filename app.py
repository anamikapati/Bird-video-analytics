from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
import cv2
import time

from detector import BirdDetector
from tracker import BirdTracker
from weight import compute_weight_index

app = FastAPI(title="Bird Counting API")

detector = BirdDetector()
tracker = BirdTracker()


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/analyze_video")
def analyze_video(file: UploadFile = File(...)):
    # Save uploaded video
    os.makedirs("uploads", exist_ok=True)
    video_path = f"uploads/{file.filename}"

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    counts = []
    weights = []

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        timestamp = round(frame_idx / fps, 2)
        count = len(tracks)

        counts.append({
            "timestamp": timestamp,
            "count": count
        })

        if len(detections) > 0:
            frame_area = frame.shape[0] * frame.shape[1]
            avg_weight = sum(
                compute_weight_index(det[:4], frame_area)
                for det in detections
            ) / len(detections)

            weights.append({
                "timestamp": timestamp,
                "avg_weight_index": round(avg_weight, 4)
            })

        frame_idx += 1

    cap.release()

    return JSONResponse({
        "counts": counts[:50],
        "weights": weights[:50],
        "note": "Weight is a relative index. Calibration needed for grams."
    })

@app.get("/")
def root():
    return {"message": "Bird Counting API is running"}
