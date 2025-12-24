import cv2
from detector import BirdDetector
from tracker import BirdTracker

VIDEO_PATH = "sample_videos/birds.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
detector = BirdDetector()
tracker = BirdTracker()

frame_id = 0

while cap.isOpened() and frame_id < 10:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)

    print(f"Frame {frame_id}: Tracks = {len(tracks)}")

    if tracks.tracker_id is not None:
        for tid in tracks.tracker_id:
            print(f"  Track ID: {int(tid)}")

    frame_id += 1

cap.release()
