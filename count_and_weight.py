import cv2
import numpy as np
import supervision as sv
from detector import BirdDetector

VIDEO_PATH = "sample_videos/birds.mp4"

# Initialize
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

detector = BirdDetector()
tracker = sv.ByteTrack()

counts_over_time = []
weights_over_time = []

frame_idx = 0

while cap.isOpened() and frame_idx < 50:   # limit for now
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_area = h * w

    detections = detector.detect(frame)

    if len(detections) == 0:
        frame_idx += 1
        continue

    xyxy = np.array([d[:4] for d in detections], dtype=np.float32)
    conf = np.array([d[4] for d in detections], dtype=np.float32)

    sv_detections = sv.Detections(
        xyxy=xyxy,
        confidence=conf
    )

    tracked = tracker.update_with_detections(sv_detections)

    # ---- COUNT ----
    count = len(tracked)
    timestamp = frame_idx / fps
    counts_over_time.append({
        "timestamp": round(timestamp, 2),
        "count": count
    })

    # ---- WEIGHT INDEX ----
    frame_weights = []
    for box in tracked.xyxy:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        weight_index = area / frame_area
        frame_weights.append(weight_index)

    avg_weight_index = float(np.mean(frame_weights))
    weights_over_time.append({
        "timestamp": round(timestamp, 2),
        "avg_weight_index": round(avg_weight_index, 5)
    })

    frame_idx += 1

cap.release()

print("Counts (sample):", counts_over_time[:5])
print("Weights (sample):", weights_over_time[:5])
