import cv2
from detector import BirdDetector

VIDEO_PATH = "sample_videos/birds.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
detector = BirdDetector()

frame_id = 0

while cap.isOpened() and frame_id < 20:   # only first 20 frames
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    count = len(detections)

    print(f"Frame {frame_id}: Count = {count}")
    frame_id += 1

cap.release()
