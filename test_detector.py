import cv2
from detector import BirdDetector

VIDEO_PATH = "sample_videos/birds.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()

detector = BirdDetector()
detections = detector.detect(frame)

print(f"Detections found: {len(detections)}")
for d in detections:
    print(d)

cap.release()
